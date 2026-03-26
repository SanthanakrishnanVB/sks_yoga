import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import pickle
import sys
from collections import deque  # Added for sliding window

# ==========================================
# 1. MODEL DEFINITION (Unchanged)
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
        self.seq_len = 30  # Frame window size
        
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
                self.num_classes = 5 
                self.idx_to_class = {}
        else:
            self.num_classes = 5

        # Initialize Model
        adj = get_adjacency_matrix()
        self.model = GTANet(33, 3, self.num_classes, adj, self.seq_len).to(self.device)
        
        # Load Weights (Ignore errors if strict loading fails due to minor version mismatch)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)
            
        self.model.eval()
        
        # MediaPipe Setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # static_image_mode=False is crucial for video/webcam smoothing
        self.pose = self.mp_pose.Pose(static_image_mode=False, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5,
                                      model_complexity=1)

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

    def start_webcam_inference(self):
        # 0 is usually the default webcam. Change to 1 or 2 if you have external cams.
        cap = cv2.VideoCapture(0)
        
        # Sliding window buffer
        frame_buffer = deque(maxlen=self.seq_len)
        
        print(f"\n[INFO] Starting Webcam... Press 'q' to exit.")
        
        current_prediction = "Waiting..."
        confidence_score = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract Pose
            lm_data, landmarks = self.extract_landmarks(frame)
            
            # Draw Skeleton
            if landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Add current frame data to buffer
                frame_buffer.append(lm_data)
            else:
                # If pose is lost, you might want to clear buffer or keep waiting
                # keeping waiting is usually better for momentary occlusion
                pass

            # Prediction Logic
            if len(frame_buffer) == self.seq_len:
                # Convert buffer to tensor: (1, 30, 33, 3)
                input_tensor = torch.tensor(list(frame_buffer), dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                
                pred_class = self.idx_to_class.get(pred_idx.item(), f"Class {pred_idx.item()}")
                confidence_score = conf.item() * 100
                current_prediction = pred_class
                
                # Visual Indicator for high/low confidence
                color = (0, 255, 0) if confidence_score > 70 else (0, 165, 255)
            else:
                # Buffering phase
                current_prediction = f"Buffering {len(frame_buffer)}/{self.seq_len}"
                color = (0, 255, 255) # Yellow
                confidence_score = 0.0

            # UI Display
            # Background rectangle for text
            cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
            
            cv2.putText(frame, f"Asana: {current_prediction}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if len(frame_buffer) == self.seq_len:
                cv2.putText(frame, f"Conf: {confidence_score:.1f}%", (20, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Yoga Asana Predictor (Live)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Ensure these paths are correct
    MODEL_PATH = "gta_net_yogav1.pth"
    META_PATH = "processed_yoga_data.pkl" 

    try:
        predictor = YogaPredictor(MODEL_PATH, META_PATH)
        predictor.start_webcam_inference()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")