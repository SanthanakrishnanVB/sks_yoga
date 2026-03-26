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

SEQ_LENGTH = 30
NUM_NODES = 33
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard Joint Angles (Internal joint bending)
JOINTS_TO_TRACK = {
    'Left Elbow': (11, 13, 15),
    'Right Elbow': (12, 14, 16),
    'Left Knee': (23, 25, 27),
    'Right Knee': (24, 26, 28),
    'Left Shoulder': (23, 11, 13),
    'Right Shoulder': (24, 12, 14),
    'Left Hip': (11, 23, 25),
    'Right Hip': (12, 24, 26)
}

# ==========================================
# PART 1: UTILITIES
# ==========================================
def calculate_angle_3d(a, b, c):
    """Calculates angle (0-180) between three 3D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_arm_body_angle(shoulder, elbow, hip):
    """
    Calculates the angle between the Arm (Shoulder->Elbow) 
    and the Torso (Shoulder->Hip).
    0 deg = Arm down at side
    90 deg = Arm T-pose (YOUR ISSUE)
    180 deg = Arm straight up overhead
    """
    shoulder = np.array(shoulder)
    elbow = np.array(elbow)
    hip = np.array(hip)
    
    # Vector 1: Shoulder to Elbow (Arm)
    vec_arm = elbow - shoulder
    # Vector 2: Shoulder to Hip (Torso)
    vec_torso = hip - shoulder
    
    cosine_angle = np.dot(vec_arm, vec_torso) / (np.linalg.norm(vec_arm) * np.linalg.norm(vec_torso) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

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
        if not os.path.exists(train_dir):
            print(f"ERROR: Dataset folder '{train_dir}' not found.")
            return None, None

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
            nn.Conv1d(64 * num_nodes, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.att_w = nn.Linear(128, 1)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat = x.view(B * T, N, C)
        out_gcn = torch.matmul(self.adj, x_flat) 
        out_gcn = F.relu(self.joint_gcn(out_gcn))
        spatial_feat = out_gcn.view(B, T, -1).permute(0, 2, 1)
        out_tcn = self.tcn(spatial_feat).permute(0, 2, 1)
        scores = self.att_w(out_tcn).squeeze(-1)
        context = (out_tcn * F.softmax(scores, dim=1).unsqueeze(-1)).sum(dim=1)
        return self.classifier(context)

# ==========================================
# PART 3: TRAINING & LEARNING
# ==========================================
def learn_ideal_poses(train_data, train_labels, class_map):
    print("\n[Auto-Learning] Analyzing ideal angles from training data...")
    idx_to_class = {v: k for k, v in class_map.items()}
    stats = {cls: {j: [] for j in JOINTS_TO_TRACK} for cls in class_map}
    
    # Store Arm-Body Angles specifically
    arm_stats = {cls: {'Left': [], 'Right': []} for cls in class_map}

    for i in tqdm(range(len(train_data))):
        pose_name = idx_to_class[train_labels[i]]
        # Use middle frame
        frame = train_data[i][SEQ_LENGTH // 2]
        
        # Standard Joints
        for joint, (a, b, c) in JOINTS_TO_TRACK.items():
            stats[pose_name][joint].append(calculate_angle_3d(frame[a], frame[b], frame[c]))

        # NEW: Arm Abduction (Indices: 11=L_Shoulder, 13=L_Elbow, 23=L_Hip)
        l_arm_angle = calculate_arm_body_angle(frame[11], frame[13], frame[23])
        r_arm_angle = calculate_arm_body_angle(frame[12], frame[14], frame[24])
        arm_stats[pose_name]['Left'].append(l_arm_angle)
        arm_stats[pose_name]['Right'].append(r_arm_angle)

    final_refs = {}
    for cls in stats:
        final_refs[cls] = {}
        # Save standard joints
        for j_name, angles in stats[cls].items():
            if angles:
                final_refs[cls][j_name] = (np.mean(angles), np.std(angles))
        # Save Arm Positions
        if arm_stats[cls]['Left']:
            final_refs[cls]['Arm_Left'] = (np.mean(arm_stats[cls]['Left']), np.std(arm_stats[cls]['Left']))
        if arm_stats[cls]['Right']:
            final_refs[cls]['Arm_Right'] = (np.mean(arm_stats[cls]['Right']), np.std(arm_stats[cls]['Right']))
    
    with open(LEARNED_ANGLES_FILE, "wb") as f: pickle.dump(final_refs, f)
    print(f"Ideal angles saved to {LEARNED_ANGLES_FILE}")

def train_pipeline():
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.create_dataset()
    
    if datasets is None or len(datasets['train']['data']) == 0:
        print("Error: No training data found.")
        return

    # Learn Ideal Poses
    learn_ideal_poses(datasets['train']['data'], datasets['train']['labels'], class_map)

    # Train Model
    train_loader = DataLoader(YogaDataset(datasets['train']['data'], datasets['train']['labels']), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    adj = get_adjacency_matrix()
    model = GTANet(NUM_NODES, CHANNELS, len(class_map), adj).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting Training for {EPOCHS} epochs...")
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
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Acc: {correct/total*100:.2f}%")
        
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training Complete. Model Saved.")

# ==========================================
# PART 4: SYSTEM & PREDICTION
# ==========================================
class YogaSystem:
    def __init__(self):
        if not os.path.exists(PROCESSED_DATA_FILE): raise FileNotFoundError("Data not found.")
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
        else:
            print("WARNING: Model weights not found!")
            
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def analyze_user_input(self, input_path):
        print(f"Attempting to load: {input_path}")
        img = cv2.imread(input_path)
        if img is None: 
            print(f"ERROR: Could not read image at {input_path}")
            return

        display_frame = img.copy()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mp_pose.process(image_rgb)
        if not res.pose_landmarks: 
            print("ERROR: No pose detected.")
            return

        single_frame = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
        raw_data = np.tile(single_frame, (SEQ_LENGTH, 1, 1))

        # Classify
        tensor_in = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_class = self.idx_to_class[pred_idx.item()]
        confidence = conf.item() * 100
        print(f"SUCCESS: Detected {pred_class} ({confidence:.1f}%)")
        
        # Feedback Logic
        feedback_lines = []
        user_frame = raw_data[-1]
        
        # Calculate Arm Angles (0=Down, 90=Side, 180=Up)
        l_arm = calculate_arm_body_angle(user_frame[11], user_frame[13], user_frame[23])
        r_arm = calculate_arm_body_angle(user_frame[12], user_frame[14], user_frame[24])
        print(f"DEBUG ANGLES -> Left Arm: {l_arm:.1f}°, Right Arm: {r_arm:.1f}°")

        # Get References (or empty dict if missing)
        refs = self.reference_angles.get(pred_class, {})
        
        # 1. Standard Joint Checks (Only if we have training data)
        for joint, (idx_a, idx_b, idx_c) in JOINTS_TO_TRACK.items():
            if joint in refs:
                user_angle = calculate_angle_3d(user_frame[idx_a], user_frame[idx_b], user_frame[idx_c])
                target, std = refs[joint]
                if abs(user_angle - target) > (std * 2) + 15:
                    correction = "Straighten" if user_angle < target else "Bend"
                    feedback_lines.append(f"{correction} {joint}")

        # 2. VRIKSHASANA (TREE POSE) SPECIAL CHECK
        # If the detected class is Tree Pose, we FORCE the arms to be overhead (170 degrees)
        # We do this regardless of training data.
        if "vrik" in pred_class.lower() or "tree" in pred_class.lower():
            print("DEBUG: Applying Vrikshasana Rules")
            target_arm = 170 # Ideal is arms up
            
            # Check Left Arm
            if l_arm < 140: # If arm is below 140 degrees
                feedback_lines.append("Raise Left Arm Overhead")
                print(f"Correction Added: Raise Left Arm (Current: {l_arm:.1f})")
            
            # Check Right Arm
            if r_arm < 140:
                feedback_lines.append("Raise Right Arm Overhead")
                print(f"Correction Added: Raise Right Arm (Current: {r_arm:.1f})")

        # 3. General Fallback (If training data exists for arms)
        elif 'Arm_Left' in refs: 
            t_l, s_l = refs['Arm_Left']
            if abs(l_arm - t_l) > (s_l * 2) + 20:
                if l_arm < 140 and t_l > 160: feedback_lines.append("Raise Left Arm")
                elif l_arm > 160 and t_l < 100: feedback_lines.append("Lower Left Arm")

        if not feedback_lines: feedback_lines.append("Perfect Posture!")
        
        self.show_results(display_frame, pred_class, confidence, feedback_lines)

    def show_results(self, img, cls, conf, feedback):
        h, w, _ = img.shape
        # Sidebar
        cv2.rectangle(img, (0, 0), (450, h), (50, 50, 50), -1)
        
        cv2.putText(img, f"Pose: {cls}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Conf: {conf:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        y = 140
        for line in feedback:
            # Red for bad, Green for good
            color = (0, 0, 255) if "Straighten" in line or "Bend" in line or "Raise" in line or "Lower" in line else (0, 255, 0)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 30
            
        # SAVE FIRST
        output_file = "output_prediction.jpg"
        cv2.imwrite(output_file, img)
        print(f"Result saved to: {output_file}")
        
        try:
            cv2.imshow("Yoga AI", img)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display window. Check output_prediction.jpg")

if __name__ == "__main__":
    print("=== YOGA AI SYSTEM v3 ===")
    choice = input("1. Train | 2. Predict: ")
    
    if choice == '1': 
        train_pipeline()
    elif choice == '2':
        path = input("Enter image path: ").strip().strip('"')
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: File not found: {path}")
        else:
            try:
                system = YogaSystem()
                system.analyze_user_input(path)
            except Exception as e:
                print(f"Unexpected Error: {e}")
                traceback.print_exc()