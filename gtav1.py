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
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "dataset"
PROCESSED_DATA_FILE = "processed_yoga_data.pkl"
LEARNED_ANGLES_FILE = "ideal_pose_angles.pkl"
MODEL_SAVE_PATH = "gta_net_yoga_final.pth"

SEQ_LENGTH = 30         # Frames to analyze
NUM_NODES = 33          # MediaPipe joints
CHANNELS = 3            # x, y, z
BATCH_SIZE = 16
EPOCHS = 50             # Increase for better accuracy
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Joint Definitions for Angle Calculation (MediaPipe Indices)
JOINTS_TO_TRACK = {
    'Left Elbow': (11, 13, 15),      # Shoulder, Elbow, Wrist
    'Right Elbow': (12, 14, 16),
    'Left Knee': (23, 25, 27),       # Hip, Knee, Ankle
    'Right Knee': (24, 26, 28),
    'Left Shoulder': (23, 11, 13),   # Hip, Shoulder, Elbow
    'Right Shoulder': (24, 12, 14),
    'Left Hip': (11, 23, 25),        # Shoulder, Hip, Knee
    'Right Hip': (12, 24, 26)
}

# ==========================================
# PART 1: UTILITIES
# ==========================================
def calculate_angle_3d(a, b, c):
    """Calculates angle (0-180) between three 3D points."""
    a = np.array(a) # First
    b = np.array(b) # Mid (Vertex)
    c = np.array(c) # End
    
    # Vectors
    ba = a - b
    bc = c - b
    
    # Cosine Rule
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def get_adjacency_matrix():
    """Builds graph structure for GCN."""
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
# PART 2: DATA PROCESSING
# ==========================================
class DataPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            model_complexity=1
        )

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
        
        # Resample to SEQ_LENGTH
        data = np.array(frames_landmarks)
        if len(data) >= SEQ_LENGTH:
            indices = np.linspace(0, len(data)-1, SEQ_LENGTH).astype(int)
            data = data[indices]
        else:
            # Pad with last frame
            pad = SEQ_LENGTH - len(data)
            data = np.pad(data, ((0, pad), (0, 0), (0, 0)), mode='edge')
        return data

    def create_dataset(self):
        if os.path.exists(PROCESSED_DATA_FILE):
            print(f"Loading cached data from {PROCESSED_DATA_FILE}...")
            with open(PROCESSED_DATA_FILE, "rb") as f:
                return pickle.load(f)

        print("Processing raw video dataset... (This happens only once)")
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

        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump((datasets, class_map), f)
        
        return datasets, class_map

class YogaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# ==========================================
# PART 3: MODEL (GTA-Net)
# ==========================================
class GTANet(nn.Module):
    def __init__(self, num_nodes, in_channels, num_classes, adj_matrix):
        super(GTANet, self).__init__()
        
        # 1. Spatial (GCN)
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)
        self.joint_gcn = nn.Linear(in_channels, 64) # Simplified GCN for demo speed
        
        # 2. Temporal (TCN)
        self.tcn = nn.Sequential(
            nn.Conv1d(64 * num_nodes, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Attention
        self.att_w = nn.Linear(128, 1)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        
        # GCN: Aggregation via Adjacency
        x_flat = x.view(B * T, N, C)
        # Proper GCN: A * X * W
        out_gcn = torch.matmul(self.adj, x_flat) 
        out_gcn = self.joint_gcn(out_gcn) # (B*T, N, 64)
        out_gcn = F.relu(out_gcn)
        
        # Reshape for TCN: (B, Channels, T)
        spatial_feat = out_gcn.view(B, T, -1).permute(0, 2, 1) # (B, N*64, T)
        
        # TCN
        out_tcn = self.tcn(spatial_feat) # (B, 128, T)
        
        # Attention
        out_tcn = out_tcn.permute(0, 2, 1) # (B, T, 128)
        scores = self.att_w(out_tcn).squeeze(-1) # (B, T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        context = (out_tcn * weights).sum(dim=1) # (B, 128)
        
        return self.classifier(context)

# ==========================================
# PART 4: IDEAL POSTURE LEARNING
# ==========================================
def learn_ideal_poses(train_data, train_labels, class_map):
    """Scans training data to calculate average angles for correct poses."""
    print("\n[Auto-Learning] Analyzing training data to find ideal angles...")
    
    idx_to_class = {v: k for k, v in class_map.items()}
    stats = {cls: {j: [] for j in JOINTS_TO_TRACK} for cls in class_map}

    # Iterate through all training samples
    for i in tqdm(range(len(train_data))):
        pose_name = idx_to_class[train_labels[i]]
        # Calculate angles for the middle frame (most stable)
        frame = train_data[i][SEQ_LENGTH // 2] 
        
        for joint, (a, b, c) in JOINTS_TO_TRACK.items():
            angle = calculate_angle_3d(frame[a], frame[b], frame[c])
            stats[pose_name][joint].append(angle)

    # Save Mean and Std Dev
    final_refs = {}
    for cls, joints in stats.items():
        final_refs[cls] = {}
        for j_name, angles in joints.items():
            if angles:
                mean = np.mean(angles)
                std = np.std(angles)
                final_refs[cls][j_name] = (mean, std)
    
    with open(LEARNED_ANGLES_FILE, "wb") as f:
        pickle.dump(final_refs, f)
    print(f"[Auto-Learning] Ideal angles saved to {LEARNED_ANGLES_FILE}")

# ==========================================
# PART 5: PREDICTION & CORRECTION SYSTEM
# ==========================================
class YogaSystem:
    def __init__(self):
        # Load Resources
        if not os.path.exists(PROCESSED_DATA_FILE):
            raise FileNotFoundError("Data not found. Train the model first!")
        
        with open(PROCESSED_DATA_FILE, "rb") as f:
            _, self.class_map = pickle.load(f)
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        # Load Learned Angles
        if os.path.exists(LEARNED_ANGLES_FILE):
            with open(LEARNED_ANGLES_FILE, "rb") as f:
                self.reference_angles = pickle.load(f)
        else:
            self.reference_angles = {}

        # Load Model
        self.adj = get_adjacency_matrix()
        self.model = GTANet(NUM_NODES, CHANNELS, len(self.class_map), self.adj).to(DEVICE)
        if os.path.exists(MODEL_SAVE_PATH):
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            self.model.eval()
            print("Model loaded successfully.")
        else:
            print("Warning: Model weights not found.")

        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def analyze_user_input(self, input_path):
        # 1. Extract Skeleton
        preprocessor = DataPreprocessor()
        is_video = input_path.lower().endswith(('.mp4', '.avi'))
        
        if is_video:
            raw_data = preprocessor.process_video(input_path)
            display_frame = cv2.VideoCapture(input_path).read()[1] # Get first frame for display
        else:
            # Handle Image
            img = cv2.imread(input_path)
            display_frame = img.copy()
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.mp_pose.process(image_rgb)
            if not res.pose_landmarks: return "No Pose Detected"
            # Repeat frame for model
            single_frame = [[lm.x, lm.y, lm.z] for lm in res.pose_landmarks.landmark]
            raw_data = np.tile(single_frame, (SEQ_LENGTH, 1, 1))

        if raw_data is None: return "Could not process input."

        # 2. Classify
        tensor_in = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        pred_class = self.idx_to_class[pred_idx.item()]
        confidence = conf.item() * 100
        
        # 3. Generate Feedback
        feedback_lines = []
        user_frame = raw_data[-1] # Use last frame for angle check
        
        if pred_class in self.reference_angles:
            refs = self.reference_angles[pred_class]
            for joint, (idx_a, idx_b, idx_c) in JOINTS_TO_TRACK.items():
                user_angle = calculate_angle_3d(user_frame[idx_a], user_frame[idx_b], user_frame[idx_c])
                target_angle, std_dev = refs[joint]
                
                # Tolerance: 2 std devs + 15 degrees buffer
                threshold = (std_dev * 2) + 15
                
                if abs(user_angle - target_angle) > threshold:
                    correction = "Straighten" if user_angle < target_angle else "Bend"
                    feedback_lines.append(f"{correction} {joint} (You: {int(user_angle)}°, Ideal: {int(target_angle)}°)")
        
        if not feedback_lines: feedback_lines.append("Perfect Posture!")

        # 4. Visualize
        self.show_results(display_frame, pred_class, confidence, feedback_lines)

    def show_results(self, img, cls, conf, feedback):
        if img is None: return
        
        # Draw Overlay
        h, w, _ = img.shape
        # Sidebar
        cv2.rectangle(img, (0, 0), (400, h), (50, 50, 50), -1)
        
        cv2.putText(img, f"Pose: {cls}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Conf: {conf:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(img, "Corrections:", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y = 180
        for line in feedback:
            color = (100, 100, 255) if "Straighten" in line or "Bend" in line else (100, 255, 100)
            # Wrap text roughly
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 30
            
        cv2.imshow("Yoga AI Instructor", img)
        print(f"\nPrediction: {cls} ({conf:.1f}%)")
        print("Feedback:")
        for line in feedback: print(f" - {line}")
        
        print("\nPress any key in the window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ==========================================
# MAIN EXECUTION
# ==========================================
def train_pipeline():
    # 1. Process Data
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.create_dataset()
    
    if not datasets['train']['data'].any():
        print("Error: No training data found.")
        return

    # 2. Learn Ideal Postures
    learn_ideal_poses(datasets['train']['data'], datasets['train']['labels'], class_map)

    # 3. Train Model
    train_loader = DataLoader(YogaDataset(datasets['train']['data'], datasets['train']['labels']), 
                              batch_size=BATCH_SIZE, shuffle=True)
    
    adj = get_adjacency_matrix()
    model = GTANet(NUM_NODES, CHANNELS, len(class_map), adj).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting Training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
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

if __name__ == "__main__":
    print("=== YOGA AI SYSTEM (GTA-NET + POSTURE CORRECTION) ===")
    print("1. Train Model & Learn Ideal Poses")
    print("2. Predict & Correct User Input")
    
    choice = input("Select option (1/2): ")
    
    if choice == '1':
        train_pipeline()
    elif choice == '2':
        path = input("Enter path to video or image: ").strip('"')
        if os.path.exists(path):
            system = YogaSystem()
            system.analyze_user_input(path)
        else:
            print("File not found.")
    else:
        print("Invalid choice.")