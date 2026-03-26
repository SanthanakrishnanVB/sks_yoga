import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import pickle
import sys

# ==========================================
# 1. MODEL DEFINITION (Must Match Training)
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
# 2. UTILITIES
# ==========================================
def get_adjacency_matrix(num_nodes=33):
    adj = np.eye(num_nodes)
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

class YogaPredictor:
    def __init__(self, model_path, meta_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_len = 30
        
        # Load Class Mapping
        if meta_path:
            try:
                with open(meta_path, "rb") as f:
                    _, self.class_map = pickle.load(f)
                self.idx_to_class = {v: k for k, v in self.class_map.items()}
                self.num_classes = len(self.class_map)
                print(f"Loaded classes: {list(self.class_map.keys())}")
            except:
                print("Warning: Could not load class map. Using generic IDs.")
                self.num_classes = 5 # Default fallback, adjust if needed
                self.idx_to_class = {}
        else:
            self.num_classes = 5

        # Initialize Model
        adj = get_adjacency_matrix()
        self.model = GTANet(33, 3, self.num_classes, adj, self.seq_len).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    def extract_landmarks(self, frame):
        """Extracts (33, 3) landmarks from a single frame."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            data = []
            for lm in results.pose_landmarks.landmark:
                data.append([lm.x, lm.y, lm.z])
            return np.array(data), results.pose_landmarks
        return None, None

    def predict(self, input_path, is_video=True):
        input_data = []
        original_frame = None # To show output

        if is_video:
            cap = cv2.VideoCapture(input_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                lm_data, _ = self.extract_landmarks(frame)
                if lm_data is not None:
                    frames.append(lm_data)
                original_frame = frame # Keep last frame for display
            cap.release()
            
            if not frames: return "No pose detected"
            
            # Resample to 30 frames
            data = np.array(frames)
            indices = np.linspace(0, len(data)-1, self.seq_len).astype(int)
            input_data = data[indices]

        else: # Image
            frame = cv2.imread(input_path)
            if frame is None: return "Image not found"
            lm_data, landmarks = self.extract_landmarks(frame)
            if lm_data is None: return "No pose detected"
            
            # Repeat single frame 30 times to mimic a static video
            input_data = np.tile(lm_data, (self.seq_len, 1, 1))
            original_frame = frame
            
            # Draw Skeleton for visualization
            mp.solutions.drawing_utils.draw_landmarks(
                original_frame, landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        # Prepare Tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 30, 33, 3)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
        
        class_name = self.idx_to_class.get(pred_idx.item(), str(pred_idx.item()))
        confidence = conf.item() * 100
        
        # Display Result
        if original_frame is not None:
            text = f"{class_name} ({confidence:.1f}%)"
            cv2.putText(original_frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Prediction Result", original_frame)
            print(f"Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return f"{class_name} ({confidence:.2f}%)"

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Path setup
    MODEL_PATH = "gta_net_yogav1.pth"
    META_PATH = "processed_yoga_data.pkl" # Contains class names

    predictor = YogaPredictor(MODEL_PATH, META_PATH)
    
    # Get user input
    print("------------------------------------------------")
    print("GTA-Net Yoga Classifier")
    print("------------------------------------------------")
    file_path = input("Enter path to video or image file: ").strip('"')
    
    # Determine type
    is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    print(f"Processing {'video' if is_video else 'image'}...")
    try:
        result = predictor.predict(file_path, is_video)
        print(f"\n>> PREDICTION: {result}")
    except Exception as e:
        print(f"Error: {e}")