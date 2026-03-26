import os
import cv2
import torch
import pickle
import numpy as np
import math
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import mediapipe as mp
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
# PART 2: DATA & MODEL (Same as before)
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
# PART 3: ENHANCED LEARNING (Tracks Arm Position)
# ==========================================
def learn_ideal_poses(train_data, train_labels, class_map):
    print("\n[Auto-Learning] Analyzing ideal angles...")
    idx_to_class = {v: k for k, v in class_map.items()}
    stats = {cls: {j: [] for j in JOINTS_TO_TRACK} for cls in class_map}
    
    # Store Arm-Body Angles specifically
    arm_stats = {cls: {'Left': [], 'Right': []} for cls in class_map}

    for i in tqdm(range(len(train_data))):
        pose_name = idx_to_class[train_labels[i]]
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
            final_refs[cls][j_name] = (np.mean(angles), np.std(angles))
        # Save Arm Positions
        final_refs[cls]['Arm_Left'] = (np.mean(arm_stats[cls]['Left']), np.std(arm_stats[cls]['Left']))
        final_refs[cls]['Arm_Right'] = (np.mean(arm_stats[cls]['Right']), np.std(arm_stats[cls]['Right']))
    
    with open(LEARNED_ANGLES_FILE, "wb") as f: pickle.dump(final_refs, f)

# ==========================================
# PART 4: SYSTEM
# ==========================================
class YogaSystem:
    def __init__(self):
        if not os.path.exists(PROCESSED_DATA_FILE): raise FileNotFoundError("Data not found.")
        with open(PROCESSED_DATA_FILE, "rb") as f: _, self.class_map = pickle.load(f)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        if os.path.exists(LEARNED_ANGLES_FILE):
            with open(LEARNED_ANGLES_FILE, "rb") as f: self.reference_angles = pickle.load(f)
        self.adj = get_adjacency_matrix()
        self.model = GTANet(NUM_NODES, CHANNELS, len(self.class_map), self.adj).to(DEVICE)
        if os.path.exists(MODEL_SAVE_PATH): self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE)); self.model.eval()
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def analyze_user_input(self, input_path):
        preprocessor = DataPreprocessor()
        img = cv2.imread(input_path)
        if img is None: return "Error loading image"
        display_frame = img.copy()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.mp_pose.process(image_rgb)
        if not res.pose_landmarks: return "No Pose Detected"
        single_frame = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
        raw_data = np.tile(single_frame, (SEQ_LENGTH, 1, 1))

        # Classify
        tensor_in = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            conf, pred_idx = torch.max(F.softmax(self.model(tensor_in), dim=1), 1)
        pred_class = self.idx_to_class[pred_idx.item()]
        confidence = conf.item() * 100
        
        # Feedback Logic
        feedback_lines = []
        user_frame = raw_data[-1]
        
        if pred_class in self.reference_angles:
            refs = self.reference_angles[pred_class]
            
            # 1. Check Standard Joints
            for joint, (idx_a, idx_b, idx_c) in JOINTS_TO_TRACK.items():
                user_angle = calculate_angle_3d(user_frame[idx_a], user_frame[idx_b], user_frame[idx_c])
                target, std = refs[joint]
                if abs(user_angle - target) > (std * 2) + 15:
                    correction = "Straighten" if user_angle < target else "Bend"
                    feedback_lines.append(f"{correction} {joint}")

            # 2. Check Arm Position (Corrects 'Broad Open' Hands)
            # Indices: 11=L_Shldr, 13=L_Elb, 23=L_Hip
            l_arm = calculate_arm_body_angle(user_frame[11], user_frame[13], user_frame[23])
            r_arm = calculate_arm_body_angle(user_frame[12], user_frame[14], user_frame[24])
            
            # Compare Left Arm
            t_l, s_l = refs['Arm_Left']
            if abs(l_arm - t_l) > (s_l * 2) + 20: # +20 buffer
                if l_arm > 160 and t_l < 100: feedback_lines.append("Lower Left Arm") # If arm is up but should be down
                elif l_arm < 150 and t_l > 160: feedback_lines.append("Raise Left Arm Overhead")
                elif 70 < l_arm < 110 and t_l > 160: feedback_lines.append("Raise Left Arm Higher") # T-pose case

            # Compare Right Arm
            t_r, s_r = refs['Arm_Right']
            if abs(r_arm - t_r) > (s_r * 2) + 20:
                if 70 < r_arm < 110 and t_r > 160: feedback_lines.append("Raise Right Arm Higher")

        if not feedback_lines: feedback_lines.append("Perfect Posture!")
        self.show_results(display_frame, pred_class, confidence, feedback_lines)

    def show_results(self, img, cls, conf, feedback):
        h, w, _ = img.shape
        cv2.rectangle(img, (0, 0), (400, h), (50, 50, 50), -1)
        cv2.putText(img, f"Pose: {cls}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Conf: {conf:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        y = 140
        for line in feedback:
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 30
        cv2.imshow("Yoga AI", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("1. Train | 2. Predict: ")
    if choice == '1': 
        # Dummy call to training pipeline (user must implement full main loop)
        print("Run training logic here...")
    elif choice == '2':
        path = input("Path: ").strip('"')
        YogaSystem().analyze_user_input(path)