import os
import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mediapipe as mp
import sys
import traceback
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# CONFIGURATION

DATASET_PATH = "dataset"
PROCESSED_DATA_FILE = "processed_yoga_data.pkl"
LEARNED_ANGLES_FILE = "ideal_pose_angles.pkl"
MODEL_SAVE_PATH = "gta_net_yoga_final.pth"

# STRICTNESS SETTINGS 
STRICT_TOLERANCE = 22.0  # Default allowed error for most poses
SEQ_LENGTH = 30
NUM_NODES = 33
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GEOMETRY ENGINE

class GeometryEngine:
    @staticmethod
    def calculate_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def get_pose_profile(landmarks):
        frame = landmarks
        profile = {}
        
        joints = {
            'Left Elbow': (11, 13, 15),
            'Right Elbow': (12, 14, 16),
            'Left Knee': (23, 25, 27),
            'Right Knee': (24, 26, 28),
            'Left Shoulder': (23, 11, 13),
            'Right Shoulder': (24, 12, 14),
            'Left Hip': (11, 23, 25),
            'Right Hip': (12, 24, 26)
        }
        
        for name, (a, b, c) in joints.items():
            profile[name] = GeometryEngine.calculate_angle(frame[a], frame[b], frame[c])

        profile['Left Armpit'] = GeometryEngine.calculate_angle(frame[13], frame[11], frame[23])
        profile['Right Armpit'] = GeometryEngine.calculate_angle(frame[14], frame[12], frame[24])

        return profile

def get_adjacency_matrix():
    adj = np.eye(NUM_NODES)
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

# DATA & MODEL

class DataPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames_landmarks = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(image_rgb)
            if results.pose_landmarks:
                data = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                frames_landmarks.append(data)
        cap.release()
        
        if not frames_landmarks: return None
        
        data = np.array(frames_landmarks)
        if len(data) >= SEQ_LENGTH:
            indices = np.linspace(0, len(data)-1, SEQ_LENGTH).astype(int)
            data = data[indices]
        else:
            pad = SEQ_LENGTH - len(data)
            data = np.pad(data, ((0, pad), (0, 0), (0, 0)), mode='edge')
        return data

    def create_dataset(self):
        if os.path.exists(PROCESSED_DATA_FILE):
            print(f"Loading cached data from {PROCESSED_DATA_FILE}...")
            with open(PROCESSED_DATA_FILE, "rb") as f: return pickle.load(f)
        
        print("Processing raw video dataset...")
        datasets = {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}}
        
        train_dir = os.path.join(DATASET_PATH, "train")
        if not os.path.exists(train_dir): return None, None

        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_map = {cls: i for i, cls in enumerate(classes)}

        for split in ["train", "test"]:
            path = os.path.join(DATASET_PATH, split)
            for cls in classes:
                cls_path = os.path.join(path, cls)
                if not os.path.exists(cls_path): continue
                print(f"Processing {split}/{cls}...")
                for vid in tqdm(os.listdir(cls_path)):
                    if vid.lower().endswith(('.mp4', '.avi', '.mov')):
                        lm = self.process_video(os.path.join(cls_path, vid))
                        if lm is not None:
                            datasets[split]["data"].append(lm)
                            datasets[split]["labels"].append(class_map[cls])
                            
            datasets[split]["data"] = np.array(datasets[split]["data"])
            datasets[split]["labels"] = np.array(datasets[split]["labels"])

        with open(PROCESSED_DATA_FILE, "wb") as f: pickle.dump((datasets, class_map), f)
        return datasets, class_map

class YogaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class GTANet(nn.Module):
    def __init__(self, num_nodes, in_channels, num_classes, adj_matrix):
        super(GTANet, self).__init__()
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)
        self.joint_gcn = nn.Linear(in_channels, 64)
        self.tcn = nn.Sequential(
            nn.Conv1d(64 * num_nodes, 128, kernel_size=5, padding=2), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.2)
        )
        self.att_w = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat = x.view(B * T, N, C)
        out_gcn = F.relu(self.joint_gcn(torch.matmul(self.adj, x_flat)))
        spatial_feat = out_gcn.view(B, T, -1).permute(0, 2, 1)
        out_tcn = self.tcn(spatial_feat).permute(0, 2, 1)
        scores = self.att_w(out_tcn).squeeze(-1)
        context = (out_tcn * F.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.classifier(context)


#TRAINING LOGIC

def learn_ideal_poses_logic(datasets, class_map):
    print("Learning ideal angles...")
    train_data = datasets['train']['data']
    train_labels = datasets['train']['labels']
    idx_to_class = {v: k for k, v in class_map.items()}
    stats = {cls: {} for cls in class_map}

    for i in tqdm(range(len(train_data))):
        pose_name = idx_to_class[train_labels[i]]
        frame = train_data[i][SEQ_LENGTH // 2]
        profile = GeometryEngine.get_pose_profile(frame)
        for joint_name, angle in profile.items():
            if joint_name not in stats[pose_name]: stats[pose_name][joint_name] = []
            stats[pose_name][joint_name].append(angle)

    final_refs = {}
    for cls in stats:
        final_refs[cls] = {}
        for joint, angles in stats[cls].items():
            if len(angles) > 0:
                final_refs[cls][joint] = float(np.mean(angles))
            
    with open(LEARNED_ANGLES_FILE, "wb") as f: pickle.dump(final_refs, f)
    print(f"Saved ideal angles to {LEARNED_ANGLES_FILE}")

def run_full_training():
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.create_dataset()
    if datasets is None: return

    learn_ideal_poses_logic(datasets, class_map)

    print(f"Starting Training...")
    train_loader = DataLoader(YogaDataset(datasets['train']['data'], datasets['train']['labels']), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    adj = get_adjacency_matrix()
    model = GTANet(NUM_NODES, CHANNELS, len(class_map), adj).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} - Acc: {correct/total*100:.2f}%")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training Complete.")

def run_learn_only():
    if not os.path.exists(PROCESSED_DATA_FILE): print("Run Option 1 first."); return
    with open(PROCESSED_DATA_FILE, "rb") as f: datasets, class_map = pickle.load(f)
    learn_ideal_poses_logic(datasets, class_map)


#SYSTEM & PREDICTION

class YogaSystem:
    def __init__(self):
        if not os.path.exists(PROCESSED_DATA_FILE): raise FileNotFoundError("Run training first.")
        with open(PROCESSED_DATA_FILE, "rb") as f: _, self.class_map = pickle.load(f)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        if os.path.exists(LEARNED_ANGLES_FILE):
            with open(LEARNED_ANGLES_FILE, "rb") as f: self.reference_angles = pickle.load(f)
        else:
            self.reference_angles = {}

        self.adj = get_adjacency_matrix()
        self.model = GTANet(NUM_NODES, CHANNELS, len(self.class_map), self.adj).to(DEVICE)
        if os.path.exists(MODEL_SAVE_PATH): 
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            self.model.eval()
            
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def check_smart_mirror_logic(self, pose_name, profile):
        pose = pose_name.lower()
        ignore_joints = []
        is_correct = False

        if "vrikshasana" in pose or "tree" in pose:
            l_knee = profile.get('Left Knee', 180)
            r_knee = profile.get('Right Knee', 180)
            
            if abs(l_knee - r_knee) > 30: 
                standing_leg = max(l_knee, r_knee)
                bent_leg = min(l_knee, r_knee)     
                
                if standing_leg > 100 and bent_leg < 130:
                    ignore_joints += ["Left Knee", "Right Knee", "Left Hip", "Right Hip"]
                    is_correct = True
            
            l_arm = profile.get('Left Armpit', 0)
            r_arm = profile.get('Right Armpit', 0)
            l_elbow = profile.get('Left Elbow', 180)
            r_elbow = profile.get('Right Elbow', 180)
            
            overhead = (l_arm > 130 and r_arm > 130) and (l_elbow > 130 and r_elbow > 130)
            namaste = (20 < l_arm < 100 and 20 < r_arm < 100) and (l_elbow < 110 and r_elbow < 110)
            
            if overhead or namaste:
                ignore_joints += ["Left Armpit", "Right Armpit", "Left Elbow", "Right Elbow", "Left Shoulder", "Right Shoulder"]

        return is_correct, ignore_joints

    def is_valid_variation(self, pose_name, joint, user_angle, ideal_angle):
        pose = pose_name.lower()
        
        # TADASANA (MOUNTAIN POSE)
        if "tadasan" in pose or "mountain" in pose:
            if "Elbow" in joint and user_angle > 110: return True
            if "Knee" in joint and user_angle > 140: return True

        # VRIKSHASANA (TREE POSE) 
        if "vrik" in pose or "tree" in pose:
            if "Knee" in joint and (user_angle < 100 and ideal_angle < 100): return True 
            if "Hip" in joint and (user_angle < 100 and ideal_angle < 100): return True
            if "Elbow" in joint and user_angle > 110: return True

        # PADMASANA (LOTUS POSE)
        if "padma" in pose or "padam" in pose or "lotus" in pose:
            if "Knee" in joint and user_angle < ideal_angle: return True
            if "Hip" in joint and user_angle < ideal_angle: return True
            
        return False

    def analyze_user_input(self, input_path):
        """Processes a single static image."""
        print(f"Loading: {input_path}")
        img = cv2.imread(input_path)
        if img is None: print("Error: Image not found"); return
        
        pred_class, conf, feedback = self._process_frame_logic(img)
        self.show_results(img, pred_class, conf, feedback)

    def analyze_webcam(self, video_path=None):
        """Opens webcam OR video file and predicts yoga poses lively frame-by-frame."""
        if video_path:
            print(f"\n Analyzing Video File: {video_path}")
            cap = cv2.VideoCapture(video_path)
        else:
            print("\n" + "="*50)
            print(" LIVE WEBCAM MODE STARTED")
            print("Please step back so your full body is in the frame.")
            print("To STOP: Click the video window and press 'q' or 'ESC'.")
            print("Fallback: Press 'Ctrl+C' in this terminal.")
            print("="*50 + "\n")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("❌ CRITICAL ERROR: Could not open video source.")
            return

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if video_path is None:
                    frame = cv2.flip(frame, 1)
                    
                display_frame = frame.copy()

                pred_class, conf, feedback, landmarks = self._process_frame_logic(frame, return_landmarks=True)

                if landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        display_frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )

                self.show_live_results(display_frame, pred_class, conf, feedback)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("Exiting Video Mode...")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Force quit detected from terminal. Shutting down safely...")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_frame_logic(self, img, return_landmarks=False):
        """Core logic separated so it can be used for both Images and Webcam"""
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mp_pose.process(image_rgb)
        
        if not res.pose_landmarks: 
            if return_landmarks: return "No Pose Detected", 0.0, ["Please step into the frame"], None
            return "No Pose Detected", 0.0, ["Please step into the frame"]

        single_frame = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
        raw_data = np.tile(single_frame, (SEQ_LENGTH, 1, 1))

        # AI Classify
        tensor_in = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor_in)
            conf, pred_idx = torch.max(F.softmax(logits, dim=1), 1)
        pred_class = self.idx_to_class[pred_idx.item()]
        confidence_val = conf.item() * 100
        
        # Geometry Override
        profile = GeometryEngine.get_pose_profile(raw_data[-1])
        left_knee_angle = profile.get('Left Knee', 180)
        right_knee_angle = profile.get('Right Knee', 180)
        
        if (left_knee_angle < 110 or right_knee_angle < 110) and "Tadasana" in pred_class:
            for name in self.class_map.keys():
                if "Vrik" in name or "vrik" in name:
                    pred_class = name
                    confidence_val = 99.9
                    break
        
        # Smart Feedback Logic
        feedback = []
        _, ignore_joints = self.check_smart_mirror_logic(pred_class, profile)
        
        pose_lower = pred_class.lower()
        current_tolerance = STRICT_TOLERANCE
        
        #STRICT PADMASANA LOGIC 
        if "padam" in pose_lower or "padma" in pose_lower or "lotus" in pose_lower:
            current_tolerance = 15.0 # Keep strict tolerance for angles
            
          
            l_knee_y = single_frame[25][1] # Left Knee Y
            r_knee_y = single_frame[26][1] # Right Knee Y
            l_ankle_y = single_frame[27][1] # Left Ankle Y
            r_ankle_y = single_frame[28][1] # Right Ankle Y
            
            # angle over knees
            left_foot_up = l_ankle_y <= (r_knee_y + 0.05)
            right_foot_up = r_ankle_y <= (l_knee_y + 0.05)

            if not (left_foot_up and right_foot_up):
                 feedback.append("Lift feet onto opposite thighs (Full Lotus)")
                 ignore_joints += ["Left Knee", "Right Knee", "Left Hip", "Right Hip"] 
        
        if pred_class in self.reference_angles:
            refs = self.reference_angles[pred_class]
            
            for joint, user_angle in profile.items():
                if joint in ignore_joints: continue 

                if joint in refs:
                    target = refs[joint]
                    if isinstance(target, (list, tuple, np.ndarray)): target = float(target[0])
                    else: target = float(target)

                    diff = abs(user_angle - target)
                    
                    if diff > current_tolerance:
                        if ("padma" in pose_lower or "padam" in pose_lower) and ("Knee" in joint or "Hip" in joint):
                            if user_angle < target: continue 
                                
                        if self.is_valid_variation(pred_class, joint, user_angle, target): continue
                            
                        if "Armpit" in joint:
                            action = "Raise" if user_angle < target else "Lower"
                            joint_name = joint.replace("Armpit", "Arm")
                        else:
                            action = "Straighten" if user_angle < target else "Bend"
                            joint_name = joint
                            
                        feedback.append(f"{action} {joint_name} (You:{int(user_angle)}, Ideal:{int(target)})")

        if not feedback: feedback.append("Perfect Posture!")
        
        if return_landmarks:
            return pred_class, confidence_val, feedback, res.pose_landmarks
        return pred_class, confidence_val, feedback

   

    def show_results(self, img, cls, conf, feedback):
        h, w, _ = img.shape
        line_spacing = 40
        start_y = 120
        required_height = start_y + 60 + (len(feedback) * line_spacing) + 50
        
        canvas_h = max(h, required_height)
        canvas_h = max(canvas_h, 500)
        
        sidebar_w = 600
        # Light Mode
        sidebar = np.ones((int(canvas_h), sidebar_w, 3), dtype=np.uint8) * 255 
        
        # Header 
        cv2.putText(sidebar, f"Pose: {cls}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        # Subheader 
        cv2.putText(sidebar, f"Confidence: {conf:.1f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)
        
        y = start_y + 60
        if not feedback or feedback[0] == "Perfect Posture!":
            # Success text (Green)
            cv2.putText(sidebar, "Perfect Posture!", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 150, 0), 2)
        else:
            # Alert Header (Orange)
            cv2.putText(sidebar, "Corrections Needed:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
            y += 45
            for line in feedback:
                cv2.putText(sidebar, f"- {line}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y += line_spacing
                
        aspect_ratio = w / h
        new_w = int(canvas_h * aspect_ratio)
        resized_img = cv2.resize(img, (new_w, int(canvas_h)))
        
        final_output = cv2.hconcat([sidebar, resized_img])
        
        # THE FIX: Return the image directly to Streamlit!
        return final_output


    # WEBCAM
# WEBCAM & VIDEO
    def show_live_results(self, img, cls, conf, feedback):
        h, w, _ = img.shape
        
        # 1. Create a bottom bar instead of a side bar
        bar_h = 160  # Height of the info box
        bottom_bar = np.ones((bar_h, w, 3), dtype=np.uint8) * 250 # Light gray background
        
        # 2. Add Header Info
        cv2.putText(bottom_bar, f"Pose: {cls}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(bottom_bar, f"Conf: {conf:.1f}%", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # 3. Add Feedback Text
        y = 105
        if not feedback or feedback[0] == "Perfect Posture!" or feedback[0] == "Please step into the frame":
            color = (0, 150, 0) if feedback and feedback[0] == "Perfect Posture!" else (0, 100, 255)
            msg = feedback[0] if feedback else "Waiting..."
            cv2.putText(bottom_bar, msg, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(bottom_bar, "Corrections Needed:", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            y += 25
            for line in feedback:
                if y > bar_h - 10:
                    cv2.putText(bottom_bar, "... and more", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    break
                cv2.putText(bottom_bar, f"- {line}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y += 22
                
        # 4. THE FIX: Vertically stack the video frame and the bottom info bar
        final_output = cv2.vconcat([img, bottom_bar])
        
        return final_output

if __name__ == "__main__":
    print("=== YOGA AI (V5 FINAL - STRICT PADMASANA & LIGHT UI) ===")
    print("1. Full Training")
    print("2. Update Ideal Angles")
    print("3. Predict (Static Image)")
    print("4. Predict (Live Webcam)")
    print("5. Predict (Video File)")
    
    choice = input("Select: ").strip()
    try:
        if choice == '1': run_full_training()
        elif choice == '2': run_learn_only()
        elif choice == '3':
            path = input("Image Path: ").strip().strip('"')
            if os.path.exists(path): YogaSystem().analyze_user_input(path)
            else: print("File not found.")
        elif choice == '4':
            YogaSystem().analyze_webcam()
        elif choice == '5':
            path = input("Video Path (e.g. video.mp4): ").strip().strip('"')
            if os.path.exists(path): YogaSystem().analyze_webcam(video_path=path)
            else: print("File not found.")
    except Exception as e:
        traceback.print_exc()