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

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "dataset"
PROCESSED_DATA_FILE = "processed_yoga_data.pkl"
LEARNED_ANGLES_FILE = "ideal_pose_angles.pkl"
MODEL_SAVE_PATH = "gta_net_yoga_final.pth"

SEQ_LENGTH = 30         # Frames to sample per video
NUM_NODES = 33          # MediaPipe joints
CHANNELS = 3            # x, y, z coordinates
BATCH_SIZE = 16
EPOCHS = 50             # Training cycles
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PART 1: GEOMETRY ENGINE (The Math Logic)
# ==========================================
class GeometryEngine:
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates 3D angle between three points (a-b-c)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def get_pose_profile(landmarks):
        """
        Extracts a dictionary of ALL important angles from a skeleton.
        This includes Armpit angles (crucial for Tree Pose arms).
        """
        frame = landmarks
        profile = {}
        
        # 1. Standard Limbs (Internal Angles)
        # Format: 'Name': (Point A, Vertex B, Point C)
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

        # 2. Armpit Angles (Arm vs Torso) 
        # Measures how high the arm is raised (0=Down, 90=Side, 180=Up)
        # Left Armpit: Shoulder(11) -> Elbow(13) vs Shoulder(11) -> Hip(23)
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

# ==========================================
# PART 2: DATA & MODEL
# ==========================================
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
        # Resample to fixed length
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
        
        print("Processing raw video dataset... This may take a while.")
        datasets = {"train": {"data": [], "labels": []}, "test": {"data": [], "labels": []}}
        
        train_dir = os.path.join(DATASET_PATH, "train")
        if not os.path.exists(train_dir):
            print(f"ERROR: '{train_dir}' not found. Check folder structure.")
            return None, None

        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_map = {cls: i for i, cls in enumerate(classes)}

        for split in ["train", "test"]:
            path = os.path.join(DATASET_PATH, split)
            if not os.path.exists(path): continue
            
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

# ==========================================
# PART 3: TRAINING & LEARNING FUNCTIONS
# ==========================================
def learn_ideal_poses_logic(datasets, class_map):
    """Scans dataset and updates the 'Gold Standard' file."""
    print("Learning ideal angles from training data...")
    train_data = datasets['train']['data']
    train_labels = datasets['train']['labels']
    idx_to_class = {v: k for k, v in class_map.items()}

    # Storage
    stats = {cls: {} for cls in class_map}

    for i in tqdm(range(len(train_data))):
        pose_name = idx_to_class[train_labels[i]]
        frame = train_data[i][SEQ_LENGTH // 2] # Middle frame is usually most stable
        
        # Get Universal Profile (Calculates ALL joints including armpits)
        profile = GeometryEngine.get_pose_profile(frame)
        
        for joint_name, angle in profile.items():
            if joint_name not in stats[pose_name]: stats[pose_name][joint_name] = []
            stats[pose_name][joint_name].append(angle)

    # Save Averages
    final_refs = {}
    for cls in stats:
        final_refs[cls] = {}
        for joint, angles in stats[cls].items():
            if len(angles) > 0:
                final_refs[cls][joint] = (np.mean(angles), np.std(angles))
            
    with open(LEARNED_ANGLES_FILE, "wb") as f: pickle.dump(final_refs, f)
    print(f"Success! Ideal angles saved to {LEARNED_ANGLES_FILE}")

def run_full_training():
    # 1. Process Data
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.create_dataset()
    
    if datasets is None or len(datasets['train']['data']) == 0:
        print("CRITICAL ERROR: No data found. Cannot train.")
        return

    # 2. Learn Ideal Angles
    learn_ideal_poses_logic(datasets, class_map)

    # 3. Train Neural Network
    print(f"Starting Training for {EPOCHS} epochs...")
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
    print("Training Complete. Model Saved.")

def run_learn_only():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("Error: No processed data found. Run Option 1 first.")
        return
    with open(PROCESSED_DATA_FILE, "rb") as f: datasets, class_map = pickle.load(f)
    learn_ideal_poses_logic(datasets, class_map)

# ==========================================
# PART 4: SYSTEM & PREDICTION
# ==========================================
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

    def analyze_user_input(self, input_path):
        print(f"Loading: {input_path}")
        img = cv2.imread(input_path)
        if img is None: print("Error: Image not found"); return

        display_frame = img.copy()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mp_pose.process(image_rgb)
        if not res.pose_landmarks: print("Error: No pose detected"); return

        single_frame = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
        raw_data = np.tile(single_frame, (SEQ_LENGTH, 1, 1))

        # Classify
        tensor_in = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor_in)
            conf, pred_idx = torch.max(F.softmax(logits, dim=1), 1)
        pred_class = self.idx_to_class[pred_idx.item()]
        
        print(f"Detected: {pred_class} ({conf.item()*100:.1f}%)")
        print(f"Debug: Looking for class '{pred_class}' in reference file...")

        if pred_class not in self.reference_angles:
            print(f"Debug: Class '{pred_class}' NOT found in learned angles. Cannot correct.")
        
        # Universal Feedback Logic
        feedback = []
        user_profile = GeometryEngine.get_pose_profile(raw_data[-1])
        
        if pred_class in self.reference_angles:
            refs = self.reference_angles[pred_class]
            
            for joint, user_angle in user_profile.items():
                if joint in refs:
                    target, std = refs[joint]
                    
                    # Debug print for Vrikshasana Arms
                    if "Armpit" in joint:
                        print(f"Debug {joint}: You={user_angle:.1f}, Ideal={target:.1f}, Std={std:.1f}")

                    # Universal Threshold
                    threshold = (std * 2) + 15
                    
                    if abs(user_angle - target) > threshold:
                        if "Armpit" in joint:
                            action = "Raise" if user_angle < target else "Lower"
                            joint_name = joint.replace("Armpit", "Arm") 
                        else:
                            action = "Straighten" if user_angle < target else "Bend"
                            joint_name = joint
                            
                        feedback.append(f"{action} {joint_name} (You:{int(user_angle)}°, Ideal:{int(target)}°)")

        if not feedback: feedback.append("Perfect Posture!")
        self.show_results(display_frame, pred_class, conf.item()*100, feedback)

    def show_results(self, img, cls, conf, feedback):
        h, w, _ = img.shape
        cv2.rectangle(img, (0, 0), (450, h), (40, 40, 40), -1)
        cv2.putText(img, f"Pose: {cls}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Conf: {conf:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        y = 140
        for line in feedback:
            color = (0, 0, 255) if any(x in line for x in ["Straighten", "Bend", "Raise", "Lower"]) else (0, 255, 0)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 30
            
        cv2.imwrite("output_final.jpg", img)
        print("Saved to output_final.jpg")
        try: cv2.imshow("Yoga AI", img); cv2.waitKey(0); cv2.destroyAllWindows()
        except: pass

# ==========================================
# MAIN MENU
# ==========================================
if __name__ == "__main__":
    print("=== UNIVERSAL YOGA AI ===")
    print("1. Full Training (Process Data + Train Model + Learn Angles)")
    print("2. Learn Ideal Angles ONLY (Fast Update)")
    print("3. Predict")
    
    choice = input("Select: ").strip()
    
    try:
        if choice == '1': 
            run_full_training()
        elif choice == '2':
            run_learn_only()
        elif choice == '3':
            path = input("Image Path: ").strip().strip('"')
            if os.path.exists(path):
                YogaSystem().analyze_user_input(path)
            else:
                print("Error: File not found.")
        else:
            print("Invalid selection.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()