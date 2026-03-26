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


# CONFIGURATION

DATASET_PATH = "dataset"
PROCESSED_DATA_FILE = "processed_yoga_data_v2.pkl"
MODEL_SAVE_PATH = "gta_net_two_stream.pth"
SEQ_LENGTH = 30         # Time frames (T)
NUM_NODES = 33          # Landmarks (N)
CHANNELS = 3            # (x, y, z)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Connections
POSE_CONNECTIONS = sorted(list(mp.solutions.pose.POSE_CONNECTIONS))


#  DATA PREPROCESSING

class DataPreprocessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    def get_video_data(self, video_path, target_length):
        cap = cv2.VideoCapture(video_path)
        joints_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Pose Keypoints (33 x 3)
                frame_joints = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                joints_list.append(frame_joints)
        cap.release()
        
        if len(joints_list) < 5: return None
        
        joints = np.array(joints_list)
        if len(joints) >= target_length:
            indices = np.linspace(0, len(joints)-1, target_length).astype(int)
            joints = joints[indices]
        else:
            pad_width = target_length - len(joints)
            joints = np.pad(joints, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
            
        # Bone Feature Matrix 
        bones = np.zeros_like(joints)
        for start, end in POSE_CONNECTIONS:
            bones[:, end, :] = joints[:, end, :] - joints[:, start, :]
            
        return joints, bones 

    def process_and_save(self):
        if os.path.exists(PROCESSED_DATA_FILE):
            with open(PROCESSED_DATA_FILE, "rb") as f:
                return pickle.load(f)

        datasets = {"train": {"joints": [], "bones": [], "labels": []}, 
                    "test": {"joints": [], "bones": [], "labels": []}}
        
        train_dir = os.path.join(DATASET_PATH, "train")
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        class_map = {cls: i for i, cls in enumerate(classes)}

        for split in ["train", "test"]:
            split_dir = os.path.join(DATASET_PATH, split)
            for cls in classes:
                cls_path = os.path.join(split_dir, cls)
                if not os.path.exists(cls_path): continue
                print(f"Processing {split}/{cls}...")
                for vid in tqdm(os.listdir(cls_path)):
                    if not vid.endswith(('.mp4', '.avi', '.mov')): continue
                    res = self.get_video_data(os.path.join(cls_path, vid), SEQ_LENGTH)
                    if res:
                        datasets[split]["joints"].append(res[0])
                        datasets[split]["bones"].append(res[1])
                        datasets[split]["labels"].append(class_map[cls])
            
            for key in ["joints", "bones", "labels"]:
                datasets[split][key] = np.array(datasets[split][key])

        with open(PROCESSED_DATA_FILE, "wb") as f:
            pickle.dump((datasets, class_map), f)
        return datasets, class_map


#  MODEL ARCHITECTURE

class YogaDataset(Dataset):
    def __init__(self, joints, bones, labels):
        self.joints = torch.tensor(joints, dtype=torch.float32)
        self.bones = torch.tensor(bones, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.joints[idx], self.bones[idx], self.labels[idx]

def get_adjacency_matrix():
    # Skeleton Graph 
    adj = np.eye(NUM_NODES)
    for i, j in POSE_CONNECTIONS:
        adj[i, j] = 1; adj[j, i] = 1
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = np.diag(d_inv_sqrt)
    return d_mat.dot(adj).dot(d_mat)

class GCNLayer(nn.Module):
    def __init__(self, in_c, out_c, adj):
        super().__init__()
        self.adj = adj
        self.linear = nn.Linear(in_c, out_c)
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(B * T, N, C)
        out = self.linear(torch.matmul(self.adj, x))
        return out.view(B, T, N, -1)

class HierarchicalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # Linear layer 
        self.query = nn.Linear(in_dim, 1)
    def forward(self, x):
        # Squeeze 
        raw_scores = self.query(x).squeeze(-1)
        # Softmax Activation
        attn_weights = F.softmax(raw_scores, dim=1).unsqueeze(-1)
        # Weighted 
        return (x * attn_weights).sum(dim=1)

class GTANet(nn.Module):
    def __init__(self, num_classes, adj_matrix):
        super().__init__()
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)
        
      
        # GCN Layer 1 (3 -> 128)
        self.joint_gcn1 = GCNLayer(3, 128, self.adj)
        self.bone_gcn1 = GCNLayer(3, 128, self.adj)
        
        # GCN Layer 2 (128 -> 128)
        self.joint_gcn2 = GCNLayer(128, 128, self.adj)
        self.bone_gcn2 = GCNLayer(128, 128, self.adj)
        
        # TCN
        # 1D Temporal Convolution (3 x 256)  
        self.tcn = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            #Relu Activation
            nn.ReLU()
        )
        
        # ATTENTION MODULE 
        self.attention = HierarchicalAttention(128)
        
        # CLASSIFIER HEAD 
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, joints, bones):
      
        
        # Spactial Streams
        j_feat = F.relu(self.joint_gcn1(joints)) 
        b_feat = F.relu(self.bone_gcn1(bones))   
        
        # Spatial Streams 
        j_feat = F.relu(self.joint_gcn2(j_feat))
        b_feat = F.relu(self.bone_gcn2(b_feat))
        
        #Feature Fusion  
        combined = torch.cat([j_feat, b_feat], dim=-1) # (Batch, T, 33, 256)
        spatial_pooled = combined.mean(dim=2)         
        
        # Temporal Module
        x = spatial_pooled.permute(0, 2, 1)            # Permutation (Batch, 256, T)
        t_feat = self.tcn(x)                           # Temporal Convolution & Relu
        t_feat = t_feat.permute(0, 2, 1)               # Permutation Back (Batch, T, 128)
        
        # Hierarchical Attention
        context = self.attention(t_feat)               # Weighted Summation (Batch, 128)
        
        # Classification
        return self.classifier(context)


# MAIN EXECUTION & EVALUATION

def main():
    preprocessor = DataPreprocessor()
    print("Initializing Data Preprocessing...")
    datasets, class_map = preprocessor.process_and_save()
    
    train_loader = DataLoader(YogaDataset(datasets['train']['joints'], datasets['train']['bones'], datasets['train']['labels']), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(YogaDataset(datasets['test']['joints'], datasets['test']['bones'], datasets['test']['labels']), batch_size=BATCH_SIZE, shuffle=False)

    model = GTANet(len(class_map), get_adjacency_matrix()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining on {device.type.upper()}...")
    loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        loss_val = 0
        for j, b, l in train_loader:
            j, b, l = j.to(device), b.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(j, b)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            
        avg_loss = loss_val/len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # EVALUATION PHASE 
    print("\nEvaluating Model on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for j, b, l in test_loader:
            j, b = j.to(device), b.to(device)
            outputs = model(j, b)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(l.numpy())

    reverse_class_map = {v: k for k, v in class_map.items()}
    target_names = [reverse_class_map[i] for i in range(len(class_map))]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # VISUALIZATION 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), loss_history, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    print("Loss curve saved as 'training_loss_curve.png'")

if __name__ == "__main__":
    main()