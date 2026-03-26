import os
import cv2
import math
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
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = "dataset"
PROCESSED_DATA_FILE = "processed_yoga_data.pkl"
MODEL_SAVE_PATH = "gta_net_yogav1.pth"
SEQ_LENGTH = 30         # Temporal dimension for the model (frames)
CHUNK_SIZE = 45         # Window size for polling evaluation (frames)
NUM_NODES = 33          # MediaPipe Pose landmarks
CHANNELS = 3            # (x, y, z) coordinates
BATCH_SIZE = 16
EPOCHS = 50
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
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            if results.pose_landmarks:
                frame_data = []
                for lm in results.pose_landmarks.landmark:
                    frame_data.append([lm.x, lm.y, lm.z])
                frames_landmarks.append(frame_data)
        
        cap.release()
        
        if not frames_landmarks: return None
        
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
        
        train_dir = os.path.join(DATASET_PATH, "train")
        if not os.path.exists(train_dir):
            print(f"Train directory not found: {train_dir}")
            return None, None
            
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
    adj = np.eye(NUM_NODES)
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
        
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
    def __init__(self, in_channels, out_channels, adj):
        super(GraphConv, self).__init__()
        self.adj = adj
        self.W = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(B * T, N, C)
        output = self.W(torch.matmul(self.adj, x))
        return output.view(B, T, N, -1)

class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.dropout(self.conv(x)))

class HierarchicalAttention(nn.Module):
    def __init__(self, in_dim):
        super(HierarchicalAttention, self).__init__()
        self.att_w = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        scores = self.att_w(x).squeeze(-1)       
        weights = F.softmax(scores, dim=1).unsqueeze(-1) 
        context = (x * weights).sum(dim=1)       
        return context

class GTANet(nn.Module):
    def __init__(self, num_nodes, in_channels, num_classes, adj_matrix, seq_len):
        super(GTANet, self).__init__()
        
        self.adj = nn.Parameter(torch.tensor(adj_matrix, dtype=torch.float32), requires_grad=False)
        self.joint_gcn = GraphConv(in_channels, 64, self.adj)
        self.tcn = TemporalConvNet(64 * num_nodes, 128)
        self.attention = HierarchicalAttention(128)
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        B, T, N, C = x.shape
        out_joint = F.relu(self.joint_gcn(x)) 
        spatial_feat = out_joint.view(B, T, -1).permute(0, 2, 1) 
        temp_feat = self.tcn(spatial_feat) 
        temp_feat = temp_feat.permute(0, 2, 1) 
        context = self.attention(temp_feat)    
        logits = self.classifier(context)
        return logits

# ==========================================
# PART 4: POLLING EVALUATION (RAW VIDEOS)
# ==========================================
def evaluate_test_videos_polling(model, test_dir, class_map, chunk_size, seq_length, device):
    """
    Evaluates raw test videos by splitting them into chunks (e.g., 45 frames)
    and using majority voting of 30-frame sliding windows within each chunk.
    """
    print(f"\n--- Starting Polling Evaluation on Test Videos ---")
    print(f"Chunk size: {chunk_size} frames | Model sequence length: {seq_length} frames")
    
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        model_complexity=1
    )
    
    index_to_class = {v: k for k, v in class_map.items()}
    class_names_ordered = [index_to_class[i] for i in range(len(class_map))]
    
    all_true_labels = []
    all_pred_labels = []

    model.eval()
    
    for cls_name in class_names_ordered:
        cls_path = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_path): continue
            
        true_label_idx = class_map[cls_name]
        video_files = [v for v in os.listdir(cls_path) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"Evaluating {len(video_files)} videos in class: {cls_name}...")
        
        for vid in tqdm(video_files):
            vid_path = os.path.join(cls_path, vid)
            cap = cv2.VideoCapture(vid_path)
            
            frames_landmarks = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_pose.process(image_rgb)
                
                if results.pose_landmarks:
                    frame_data = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                    frames_landmarks.append(frame_data)
            cap.release()
            
            if not frames_landmarks: continue
                
            data = np.array(frames_landmarks)
            total_frames = len(data)
            num_chunks = math.ceil(total_frames / chunk_size)
            
            # Process each chunk independently
            for c in range(num_chunks):
                start_idx = c * chunk_size
                end_idx = min(start_idx + chunk_size, total_frames)
                chunk_data = data[start_idx:end_idx]
                
                windows = []
                if len(chunk_data) < seq_length:
                    # Pad if chunk is smaller than 30 frames
                    pad_width = seq_length - len(chunk_data)
                    padded_data = np.pad(chunk_data, ((0, pad_width), (0, 0), (0, 0)), mode='edge')
                    windows.append(padded_data)
                else:
                    # Slide 30-frame window across the chunk
                    for i in range(len(chunk_data) - seq_length + 1):
                        windows.append(chunk_data[i : i + seq_length])
                
                windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model(windows_tensor)
                    _, predicted_classes = torch.max(outputs, 1)
                    predictions = predicted_classes.cpu().tolist()
                
                # Mode prediction
                if predictions:
                    most_common_pred = Counter(predictions).most_common(1)[0][0]
                    all_pred_labels.append(most_common_pred)
                    all_true_labels.append(true_label_idx)

    # Calculate and display metrics
    if all_true_labels:
        print("\n=== Polling Evaluation Report ===")
        print(classification_report(all_true_labels, all_pred_labels, target_names=class_names_ordered))
        
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names_ordered, yticklabels=class_names_ordered)
        plt.title(f"Confusion Matrix - {chunk_size}-Frame Chunk Polling")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig("polling_confusion_matrix.png")
        print("Polled Confusion Matrix saved as 'polling_confusion_matrix.png'")
    else:
        print("No valid test videos were processed.")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # 1. Process Data
    preprocessor = DataPreprocessor()
    datasets, class_map = preprocessor.process_and_save()
    
    if datasets is None or len(datasets['train']['data']) == 0:
        print("Error: No data found. Check your 'dataset' folder structure.")
        return

    num_classes = len(class_map)
    index_to_class = {v: k for k, v in class_map.items()}
    class_names_ordered = [index_to_class[i] for i in range(num_classes)]
    print(f"\nClasses Detected: {class_names_ordered}")

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

    # 5. Standard Evaluation (on padded/interpolated preprocessed test set)
    print("\nStandard Evaluating Model...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nStandard Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names_ordered))

    # 6. Polling Evaluation (on raw test videos directly from directory)
    test_directory = os.path.join(DATASET_PATH, "test")
    evaluate_test_videos_polling(
        model=model, 
        test_dir=test_directory, 
        class_map=class_map, 
        chunk_size=CHUNK_SIZE, 
        seq_length=SEQ_LENGTH, 
        device=device
    )

if __name__ == "__main__":
    main()