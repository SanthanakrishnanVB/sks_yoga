import os
import cv2
import torch
import pickle
import numpy as np
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
MODEL_SAVE_PATH = "gta_net_yogav1.pth"
SEQ_LENGTH = 30         # Temporal dimension (frames)
NUM_NODES = 33          # MediaPipe Pose landmarks
CHANNELS = 3            # (x, y, z) coordinates
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PART 1: PREPROCESSING (Video -> Skeleton)
# ==========================================
class DataPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            model_complexity=1
        )

    def get_video_landmarks(self, video_path, target_length):
        cap = cv2.VideoCapture(video_path)
        frames_landmarks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                frame_data = []
                for lm in results.pose_landmarks.landmark:
                    frame_data.append([lm.x, lm.y, lm.z])
                frames_landmarks.append(frame_data)
        
        cap.release()
        
        if not frames_landmarks: return None
        
        # Temporal Processing: Resample or Pad to fixed SEQ_LENGTH
        data = np.array(frames_landmarks) # (T_original, 33, 3)
        if len(data) >= target_length:
            indices = np.linspace(0, len(data)-1, target_length).astype(int)
            data = data[indices]
        else:
            pad_width = target_length - len(data)
            data = np.pad(data, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
            
        return data # (30, 33, 3)

    def process_and_save(self):
        if os.path.exists(PROCESSED_DATA_FILE):
            print(f"Loading existing data from {PROCESSED_DATA_FILE}...")
            with open(PROCESSED_DATA_FILE, "rb") as f:
                return pickle.load(f)

        print("Processing raw video dataset... this may take time.")
        datasets = {"train": {}, "test": {}}
        class_map = {}
        
        # Build class map from train folder
        train_dir = os.path.join(DATASET_PATH, "train")
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_map = {cls: i for i, cls in enumerate(classes)}

        for split in ["train", "test"]:
            split_dir = os.path.join(DATASET_PATH, split)
            data_list, label_list = [], []
            
            for cls in classes:
                cls_path = os.path.join(split_dir, cls)
                if not os.path.exists(cls_path): continue
                
                print(f"Processing {split}/{cls}...")
                for vid in tqdm(os.listdir(cls_path)):
                    if not vid.endswith(('.mp4', '.avi', '.mov')): continue
                    
                    landmarks = self.get_video_landmarks(os.path.join(cls_path, vid), SEQ_LENGTH)
                    if landmarks is not None:
                        data_list.append(landmarks)
                        label_list.append(class_map[cls])
            
            datasets[split]["data"] = np.array(data_list)
            datasets[split]["labels"] = np.array(label_list)

        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump((datasets, class_map), f)
            
        return datasets, class_map

# ==========================================
# PART 2: DATASET LOADING
# ==========================================
class YogaDataset(Dataset):
    def __init__(self, data_array, label_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.labels = torch.tensor(label_array, dtype=torch.long)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_adjacency_matrix():
    """Builds the graph adjacency matrix for MediaPipe's 33 joints."""
    # Based on Eq. (2) and (3) in the paper for GCN normalization [cite: 251, 272]
    adj = np.eye(NUM_NODES)
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
        
    # Normalization: D^-0.5 * A * D^-0.5
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return norm_adj

# ==========================================
# PART 3: GTA-NET MODEL
# ==========================================
class GraphConv(nn.Module):
    """
    Implements Joint-GCN / Bone-GCN Layer.
    Paper Ref: Section 3.2, Eq. (1) [cite: 248]
    """
    def __init__(self, in_channels, out_channels, adj):
        super(GraphConv, self).__init__()
        self.adj = adj
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x: (Batch, Temporal, Nodes, Channels)
        B, T, N, C = x.shape
        x = x.view(B * T, N, C)
        output = self.W(torch.matmul(self.adj, x))
        return output.view(B, T, N, -1)

class TemporalConvNet(nn.Module):
    """
    Implements Temporal Convolutional Network (TCN).
    Paper Ref: Section 3.3, Eq. (8) [cite: 294, 314]
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (Batch, Channels, Temporal)
        return self.relu(self.dropout(self.conv(x)))

class HierarchicalAttention(nn.Module):
    """
    Implements Hierarchical Attention Mechanism.
    Paper Ref: Section 3.4, Eq. (11-15) [cite: 328, 334]
    """
    def __init__(self, in_dim):
        super(HierarchicalAttention, self).__init__()
        self.att_w = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        # x: (Batch, Sequence, Features)
        scores = self.att_w(x).squeeze(-1)       # (B, S)
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # (B, S, 1)
        context = (x * weights).sum(dim=1)       # Weighted Sum
        return context

class GTANet(nn.Module):
    """
    Main GTA-Net Architecture.
    Paper Ref: Section 3.1, Fig 2 [cite: 195, 196]
    """
    def __init__(self, num_nodes, in_channels, num_classes, adj_matrix, seq_len):
        super(GTANet, self).__init__()
        
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)
        
        # 1. Spatial Stream (GCN) - Joint GCN [cite: 215]
        self.joint_gcn = GraphConv(in_channels, 64, self.adj)
        
        # 2. Temporal Stream (TCN) [cite: 294]
        # Input dim is 64 * NUM_NODES because we flatten spatial features for TCN
        self.tcn = TemporalConvNet(64 * num_nodes, 128)
        
        # 3. Attention Mechanism [cite: 328]
        self.attention = HierarchicalAttention(128)
        
        # 4. Classification Head (Replacing original regression head)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, T, N, C = x.shape
        
        # Spatial Processing (GCN)
        out_joint = F.relu(self.joint_gcn(x)) # (B, T, N, 64)
        
        # Flatten spatial features: (B, T, N*64) -> Permute to (B, N*64, T) for TCN
        spatial_feat = out_joint.view(B, T, -1).permute(0, 2, 1) 
        
        # Temporal Processing (TCN)
        temp_feat = self.tcn(spatial_feat) # (B, 128, T)
        
        # Attention over time steps
        temp_feat = temp_feat.permute(0, 2, 1) # (B, T, 128)
        context = self.attention(temp_feat)    # (B, 128)
        
        # Classification
        logits = self.classifier(context)
        return logits

# ==========================================
# PART 4: VISUALIZATION
# ==========================================
def visualize_3d_skeleton(data_sample, label, class_names):
    """Visualizes the first frame skeleton."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    frame = data_sample[0] # Take 1st frame
    
    xs, ys, zs = frame[:, 0], frame[:, 1], frame[:, 2]
    
    # Draw Joints
    ax.scatter(xs, ys, zs, c='red', s=20)
    
    # Draw Bones
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for start, end in connections:
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], c='blue')
        
    ax.set_title(f"3D Skeleton: {class_names[label]}")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    # Invert Y/Z for better visualization often required with MediaPipe coords
    ax.invert_zaxis()
    ax.invert_yaxis()
    plt.savefig("3d_skeleton_sample.png")
    print("Saved 3d_skeleton_sample.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Process Data
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.process_and_save()
    
    if len(datasets['train']['data']) == 0:
        print("Error: No data found. Check your 'dataset' folder structure.")
        return

    num_classes = len(class_map)
    class_names = list(class_map.keys())
    print(f"\nClasses Detected: {class_names}")

    # 2. Data Loaders
    train_set = YogaDataset(datasets['train']['data'], datasets['train']['labels'])
    test_set = YogaDataset(datasets['test']['data'], datasets['test']['labels'])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    adj_matrix = get_adjacency_matrix()
    model = GTANet(NUM_NODES, CHANNELS, num_classes, adj_matrix, SEQ_LENGTH).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("\nStarting Training...")
    train_losses, train_accs = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_accs.append(acc)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # 5. Evaluation & Metrics
    print("\nEvaluating Model...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Yoga Asana Classification")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    print("Confusion Matrix saved as 'confusion_matrix.png'")

    # Training Curves Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Accuracy')
    plt.title("Training Accuracy")
    plt.savefig("training_metrics.png")
    print("Training metrics saved as 'training_metrics.png'")

    # 6. Visualize Sample
    if len(test_set) > 0:
        sample_data, sample_label = test_set[0]
        visualize_3d_skeleton(sample_data.numpy(), sample_label.item(), class_names)

if __name__ == "__main__":
    main()