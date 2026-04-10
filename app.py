import streamlit as st
import cv2
import numpy as np
from PIL import Image as PILImage
import mediapipe as mp
from streamlit_option_menu import option_menu
from yoga_backend import YogaSystem
import time
import os
import tempfile
import uuid
from fpdf import FPDF
import pandas as pd
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import queue

# ==========================================
# PAGE CONFIGURATION & MOBILE-RESPONSIVE CSS
# ==========================================
st.set_page_config(page_title="SKS AI Yoga", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #F4F7F6; transition: background-color 0.3s ease; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    h1, h2, h3 { color: #2C3E50; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    .stFileUploader { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.05); }
    div.stInfo { background-color: #E8F4F8; border-left-color: #3498DB; color: #2C3E50; }
    div.stSuccess { background-color: #EBF5EA; border-left-color: #27AE60; color: #2C3E50; }
    
    /* === DESKTOP VIEWS === */
    @media (min-width: 769px) {
        [data-testid="stSidebar"] { min-width: 400px !important; max-width: 400px !important; }
        .mobile-title { font-size: 28px !important; margin-top: 15px !important; margin-bottom: 0px !important; }
        .mobile-subtitle { font-size: 14px !important; margin-top: -5px !important; }
        .logo-container { display: flex; justify-content: flex-start; }
    }
    
    /* === MOBILE VIEWS === */
    @media (max-width: 768px) {
        .mobile-title { font-size: 24px !important; margin-top: 5px !important; text-align: center !important; }
        .mobile-subtitle { font-size: 14px !important; text-align: center !important; margin-top: 0px !important; }
        .logo-container { display: flex; justify-content: center; margin-bottom: 10px; }
        .block-container { padding-top: 2rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# INITIALIZE AI MODEL & UTILS
# ==========================================
@st.cache_resource
def load_model():
    return YogaSystem()

try:
    yoga_ai = load_model()
except FileNotFoundError:
    st.error("Model files not found! Please run the training script first.")
    st.stop()

mp_pose = mp.solutions.pose
TEMP_DIR = tempfile.mkdtemp()

if 'report_data' not in st.session_state:
    st.session_state.report_data = []
if 'overall_score' not in st.session_state:
    st.session_state.overall_score = None

def reset_report():
    if st.session_state.get('webcam_toggle', False):
        st.session_state.report_data = []
        st.session_state.overall_score = None

# ==========================================
# CUSTOM DRAWING & PDF LOGIC
# ==========================================
def draw_colored_skeleton(image, landmarks, feedback):
    """Draws the skeleton using a constant color."""
    h, w, _ = image.shape
    
    # Define your constant color here (B, G, R format). 
    # (0, 255, 0) is Green. 
    constant_color = (0, 255, 0) 

    # Draw the bones (connections)
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection[0], connection[1]
        start_lm, end_lm = landmarks.landmark[start_idx], landmarks.landmark[end_idx]
        
        # Only draw if the joints are visible enough
        if start_lm.visibility < 0.3 or end_lm.visibility < 0.3: 
            continue
            
        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))
        
        cv2.line(image, start_point, end_point, constant_color, 4)
        
    # Draw the joints (landmarks)
    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility < 0.3: 
            continue
            
        point = (int(lm.x * w), int(lm.y * h))
        cv2.circle(image, point, 6, constant_color, -1)
        
    return image

def create_pdf_report(report_data, overall_score=None):
    """Generates a PDF file from the captured frames and text."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    if overall_score is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(200, 40, txt="Yoga Session Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", 'B', 18)
        
        if overall_score >= 85: pdf.set_text_color(0, 150, 0)
        elif overall_score >= 70: pdf.set_text_color(200, 150, 0)
        else: pdf.set_text_color(200, 0, 0)
            
        pdf.cell(200, 20, txt=f"Overall Session Score: {overall_score:.1f}/100", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="(Score ignores initial setup time to evaluate your true held posture)", ln=True, align='C')
    
    for item in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pose_name = item.get('pose', 'Unknown Pose')
        conf_val = item.get('conf', 0.0)
        event_label = item.get('event_label', '')
        
        header_text = f"Time: {item['time']} | Pose: {pose_name} ({conf_val:.1f}%)"
        if event_label:
            header_text += f" - {event_label}"
            
        pdf.cell(200, 10, txt=header_text, ln=True, align='C')
        
        try:
            img_pil = PILImage.open(item['img_path'])
            img_w, img_h = img_pil.size
            aspect_ratio = img_w / img_h
            
            max_pdf_width = 190
            max_pdf_height = 110
            
            if aspect_ratio > (max_pdf_width / max_pdf_height):
                pdf_w = max_pdf_width
                pdf_h = max_pdf_width / aspect_ratio
            else:
                pdf_h = max_pdf_height
                pdf_w = max_pdf_height * aspect_ratio
                
            x_offset = (210 - pdf_w) / 2
            pdf.image(item['img_path'], x=x_offset, y=30, w=pdf_w, h=pdf_h)
            
            pdf.set_xy(10, 30 + pdf_h + 10) 
            
        except Exception as e:
            pdf.set_xy(10, 40)
            pdf.cell(200, 10, txt="[Image Processing Error]", ln=True)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="AI Feedback Summary:", ln=True)
        
        pdf.set_font("Arial", size=12)
        if not item['feedback'] or item['feedback'][0] == "Perfect Posture!":
            pdf.set_text_color(0, 150, 0)
            pdf.cell(200, 10, txt="Perfect Posture! All joints aligned perfectly.", ln=True)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.set_text_color(200, 0, 0)
            for f in item['feedback']:
                pdf.multi_cell(0, 8, txt=f"- {f}")
            pdf.set_text_color(0, 0, 0)
            
    tmp_pdf_path = os.path.join(TEMP_DIR, f"report_{uuid.uuid4().hex}.pdf")
    pdf.output(tmp_pdf_path)
    
    with open(tmp_pdf_path, "rb") as f:
        pdf_bytes = f.read()
        
    try: os.remove(tmp_pdf_path)
    except: pass
    return pdf_bytes

# ==========================================
# RESPONSIVE TOP HEADER & NAVIGATION
# ==========================================
header_col1, header_col2, header_col3 = st.columns([1, 3, 8])

with header_col1:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        st.image("logo.png", width=70)
    except FileNotFoundError:
        pass
    st.markdown('</div>', unsafe_allow_html=True)

with header_col2:
    st.markdown("<h3 class='mobile-title' style='color: #2E86C1;'>SKS AI Yoga</h3>", unsafe_allow_html=True)
    st.markdown("<p class='mobile-subtitle' style='color: #7F8C8D;'>Real-time personal trainer</p>", unsafe_allow_html=True)

with header_col3:
    selected = option_menu(
        menu_title=None, 
        options=["Live Tracking", "Analyze Image", "Analyze Video", "Learning Hub"], 
        icons=["", "", "", ""], 
        default_index=0, 
        orientation="horizontal",
        styles={
            "container": {"padding": "5px!important", "background-color": "white", "border-radius": "10px", "box-shadow": "0px 4px 6px rgba(0,0,0,0.05)", "margin-top": "10px", "display": "flex", "flex-wrap": "wrap", "justify-content": "center"},
            "icon": {"display": "none"}, 
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px 5px", "padding": "10px 15px", "--hover-color": "#F2F3F4", "color": "#2C3E50", "white-space": "nowrap", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "#3498DB", "color": "white", "font-weight": "bold"},
        }
    )

st.markdown("---")

dynamic_bg_global = st.empty() 

# ==========================================
# MAIN APP LOGIC
# ==========================================

# --- 1. LIVE WEBCAM MODE (DEPLOYMENT READY) ---
if selected == "Live Tracking":
    col_video, col_dashboard = st.columns([2, 1]) 
    
    # 1. Setup a thread-safe queue to pass data from the video stream to the Streamlit UI
    if 'webrtc_queue' not in st.session_state:
        st.session_state.webrtc_queue = queue.Queue(maxsize=2)

    with col_video:
        st.markdown("### Real-Time Posture AI")
        st.info("Click 'START' below and allow camera permissions.")
        
        # 2. The Video Callback: Processes the frame and draws the skeleton
        def video_frame_callback(frame):
            # Convert browser frame to OpenCV format
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1) # Mirror the image
            
            # Run AI model
            pred_class, conf, feedback, landmarks = yoga_ai._process_frame_logic(img, return_landmarks=True)
            
            # Draw the skeleton
            if landmarks:
                img = draw_colored_skeleton(img, landmarks, feedback)
                
            # Send results to the main thread Queue for the dashboard updates
            try:
                st.session_state.webrtc_queue.put_nowait((pred_class, conf, feedback, img.copy()))
            except queue.Full:
                pass # Ignore if the queue is full to prevent video lag
                
            # Send the drawn frame back to the browser widget
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 3. Start the WebRTC Streamer
        # This replaces your old st.toggle button
        ctx = webrtc_streamer(
            key="yoga_tracker",
            video_frame_callback=video_frame_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    
    with col_dashboard:
        st.markdown("###  Live Analysis")
        # Empty placeholders for the dashboard
        ui_pose = st.empty()
        ui_conf = st.empty()
        ui_status = st.empty()
        ui_feedback = st.empty() 

    # 4. Main Thread Loop: Update the UI and save PDF data while the video is playing
    if ctx.state.playing:
        last_capture_time = time.time()
        
        while ctx.state.playing:
            try:
                # Pull the latest data from the background video thread
                pred_class, conf, feedback, frame_copy = st.session_state.webrtc_queue.get(timeout=1.0)
                
                # Update Dashboard UI Elements
                ui_pose.markdown(f"<h2 style='text-align: center; color: #3498DB; padding: 15px; background-color: white; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>🧘 {pred_class}</h2>", unsafe_allow_html=True)
                ui_conf.progress(int(conf) / 100.0, text=f"Confidence: {conf:.1f}%")
                
                # Extract frames for the PDF Report every 1 second
                current_time = time.time()
                if current_time - last_capture_time >= 1.0:
                    if feedback and "Please step into the frame" not in feedback[0]:
                        img_path = os.path.join(TEMP_DIR, f"live_{int(current_time)}.jpg")
                        cv2.imwrite(img_path, frame_copy) 
                        st.session_state.report_data.append({
                            'time': time.strftime('%H:%M:%S'),
                            'img_path': img_path,
                            'feedback': feedback,
                            'pose': pred_class,
                            'conf': conf
                        })
                    last_capture_time = current_time

                # Update Textual Feedback UI
                if feedback and feedback[0] == "Perfect Posture!":
                    ui_status.success("🟢 Perfect Posture!")
                    ui_feedback.empty()
                elif feedback and "Please step into the frame" in feedback[0]:
                    ui_status.warning("🟡 Waiting for user...")
                    ui_feedback.empty()
                else:
                    ui_status.error("🔴 Corrections needed:")
                    feedback_html = "".join([f"<li>{f}</li>" for f in feedback])
                    ui_feedback.markdown(f"<ul style='color: #E74C3C; font-weight: bold;'>{feedback_html}</ul>", unsafe_allow_html=True)
                    
            except queue.Empty:
                pass # If no new frame arrived yet, just loop again
                
    # 5. Generate PDF (Triggers automatically when the user clicks 'STOP' on the camera)
    if not ctx.state.playing and len(st.session_state.report_data) > 0:
        st.success("Session finished! Download your posture report below.")
        pdf_bytes = create_pdf_report(st.session_state.report_data)
        st.download_button(
            label="📄 Download Posture PDF Report",
            data=pdf_bytes,
            file_name="Yoga_Live_Session_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# --- 2. STATIC IMAGE MODE ---
elif selected == "Analyze Image":
    st.sidebar.markdown("### Upload Media")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('AI is calculating biomechanical joint angles...'):
            pred_class, conf, feedback, landmarks = yoga_ai._process_frame_logic(img, return_landmarks=True)
            
            if landmarks:
                img = draw_colored_skeleton(img, landmarks, feedback)
            
            col_img, col_dash = st.columns([2, 1])
            with col_img:
                st.markdown('<div style="background-color: white; padding: 10px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_dash:
                st.markdown("###  AI Evaluation")
                st.markdown(f"<h2 style='text-align: center; color: #3498DB; padding: 15px; background-color: white; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>🧘 {pred_class}</h2>", unsafe_allow_html=True)
                st.progress(int(conf) / 100.0, text=f"Confidence: {conf:.1f}%")
                
                # Show Feedback Details Instantly
                if feedback and feedback[0] == "Perfect Posture!":
                    st.success("🟢 Perfect Posture!")
                    dynamic_bg_global.markdown("<style>.stApp { background-color: #4ade80 !important; }</style>", unsafe_allow_html=True)
                else:
                    st.error("🔴 Corrections needed:")
                    for f in feedback:
                        st.write(f"- {f}")
                    dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)

            img_path = os.path.join(TEMP_DIR, "image_analysis.jpg")
            cv2.imwrite(img_path, img)
            
            report_data = [{'time': 'Image Analysis', 'img_path': img_path, 'feedback': feedback, 'pose': pred_class, 'conf': conf}]
            pdf_bytes = create_pdf_report(report_data)
            
            st.markdown("---")
            st.download_button(
                label="📄 Download Assessment PDF",
                data=pdf_bytes,
                file_name="Yoga_Image_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("### Static Form Evaluation")
        st.info("Upload an image from the sidebar to begin.")
        dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)

# --- 3. VIDEO FILE MODE WITH TWO OPTIONS ---
elif selected == "Analyze Video":
    st.sidebar.markdown("### Analysis Options")
    analysis_mode = st.sidebar.radio(
        "Select Mode:", 
        ["Frame-by-Frame (Detailed)", "Full Video (Overall Score)"],
        help="Frame-by-Frame logs data every second. Full Video calculates a final score and only extracts keyframes when your posture significantly changes."
    )
    
    st.sidebar.markdown("### Upload Media")
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    
    if uploaded_video is not None:
        if st.button("Start AI Processing", use_container_width=True):
            st.session_state.report_data = [] 
            st.session_state.overall_score = None
            
            temp_video_path = os.path.join(TEMP_DIR, "temp_uploaded_video.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.read())
            
            col_vid, col_dash = st.columns([2, 1])
            with col_vid:
                frame_window = st.empty()
                progress_bar = st.progress(0)
            
            with col_dash:
                st.markdown("###  Live Processing")
                ui_pose = st.empty()
                ui_conf = st.empty()
                ui_status = st.empty()
                ui_feedback = st.empty() # Added placeholder for real-time video text feedback
            
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or fps != fps: fps = 30 
            
            current_frame = 0
            
            # Variables for Full Video Mode Keyframe Extraction
            last_captured_conf = -100
            last_captured_pose = ""
            last_capture_sec = -10
            all_video_stats = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                pred_class, conf, feedback, landmarks = yoga_ai._process_frame_logic(frame, return_landmarks=True)
                is_perfect = (feedback and feedback[0] == "Perfect Posture!")

                if landmarks:
                    frame = draw_colored_skeleton(frame, landmarks, feedback)
                
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                ui_pose.markdown(f"<h2 style='text-align: center; color: #3498DB; padding: 15px; background-color: white; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>🧘 {pred_class}</h2>", unsafe_allow_html=True)
                ui_conf.progress(int(conf) / 100.0, text=f"Confidence: {conf:.1f}%")
                
                seconds = current_frame // int(fps)
                time_str = time.strftime('%M:%S', time.gmtime(seconds))
                
                # --- MODE 1: FRAME BY FRAME ---
                if analysis_mode == "Frame-by-Frame (Detailed)":
                    if current_frame % int(fps) == 0:
                        if feedback and "Please step into the frame" not in feedback[0]:
                            img_path = os.path.join(TEMP_DIR, f"vid_{seconds}.jpg")
                            cv2.imwrite(img_path, frame)
                            
                            st.session_state.report_data.append({
                                'time_sec': seconds,
                                'time': time_str,
                                'img_path': img_path,
                                'feedback': feedback,
                                'pose': pred_class,
                                'conf': conf,
                                'is_perfect': is_perfect
                            })

                # --- MODE 2: FULL VIDEO OVERALL SCORE (SMART KEYFRAMES) ---
                else:
                    if feedback and "Please step into the frame" not in feedback[0]:
                        all_video_stats.append({
                            'time_sec': seconds,
                            'pose': pred_class,
                            'conf': conf,
                            'is_perfect': is_perfect
                        })
                        
                        # Trigger Keyframe if Pose changes OR Confidence changes by > 15% (with 3s cooldown)
                        pose_changed = (pred_class != last_captured_pose)
                        conf_changed = abs(conf - last_captured_conf) > 15
                        cooldown_passed = (seconds - last_capture_sec) >= 3
                        
                        if (pose_changed or conf_changed) and cooldown_passed:
                            img_path = os.path.join(TEMP_DIR, f"keyframe_{seconds}.jpg")
                            cv2.imwrite(img_path, frame)
                            
                            if pose_changed:
                                event_label = f"Transitioned to {pred_class}"
                            elif conf > last_captured_conf:
                                event_label = "Form Improved"
                            else:
                                event_label = "Form Degraded"
                                
                            st.session_state.report_data.append({
                                'time_sec': seconds,
                                'time': time_str,
                                'img_path': img_path,
                                'feedback': feedback,
                                'pose': pred_class,
                                'conf': conf,
                                'is_perfect': is_perfect,
                                'event_label': event_label
                            })
                            
                            last_captured_pose = pred_class
                            last_captured_conf = conf
                            last_capture_sec = seconds

                # Background Color Update and Real-Time Text Updates
                if is_perfect:
                    ui_status.success("🟢 Perfect Posture!")
                    ui_feedback.empty()
                    dynamic_bg_global.markdown("<style>.stApp { background-color: #4ade80 !important; }</style>", unsafe_allow_html=True)
                else:
                    ui_status.error("🔴 Corrections needed:")
                    feedback_html = "".join([f"<li>{f}</li>" for f in feedback])
                    ui_feedback.markdown(f"<ul style='color: #E74C3C; font-weight: bold;'>{feedback_html}</ul>", unsafe_allow_html=True)
                    dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)
                
                current_frame += 1
                if total_frames > 0: progress_bar.progress(min(current_frame / total_frames, 1.0))
                
            cap.release()
            st.session_state['video_processed'] = True
            
            # Calculate Overall Score for Full Video Mode
            if analysis_mode == "Full Video (Overall Score)" and len(all_video_stats) > 0:
                df_stats = pd.DataFrame(all_video_stats)
                
                # Find the main pose they attempted
                main_pose = df_stats['pose'].mode()[0]
                pose_stats = df_stats[df_stats['pose'] == main_pose]
                
                # Ignore first 20% of frames (Setup time)
                setup_frames = int(len(pose_stats) * 0.20)
                stable_stats = pose_stats.iloc[setup_frames:]
                
                if len(stable_stats) > 0:
                    st.session_state.overall_score = stable_stats['conf'].mean()
                else:
                    st.session_state.overall_score = pose_stats['conf'].mean()
            
            dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)
            
        # ==========================================
        # DASHBOARD: STATS, GRAPHS & REVIEW
        # ==========================================
        if st.session_state.get('video_processed', False) and len(st.session_state.report_data) > 0:
            st.success("Video Processing Complete! Explore your analysis dashboard below.")
            df = pd.DataFrame(st.session_state.report_data)
            
            st.markdown("---")
            st.markdown("## 📊 Session Analytics Dashboard")
            
            # --- CONDITIONAL UI: Scrubber vs Gallery ---
            if analysis_mode == "Frame-by-Frame (Detailed)":
                m1, m2 = st.columns(2)
                m1.metric("Captured Frames", len(df))
                perfect_percentage = (len(df[df['is_perfect'] == True]) / len(df)) * 100
                m2.metric("Frames in Perfect Form", f"{perfect_percentage:.1f}%")
                
                # Confidence Graph
                st.markdown("### Confidence Score Over Time")
                chart_data = df[['time_sec', 'conf']].set_index('time_sec')
                st.line_chart(chart_data, color="#3498DB")

                st.markdown("### Frame-by-Frame Review (Scrubber)")
                st.info("Use the slider below to scrub through your analyzed frames.")
                
                frame_idx = st.slider("Select Captured Frame", 0, len(df)-1, 0)
                selected_frame = df.iloc[frame_idx]
                
                rev_col1, rev_col2 = st.columns([2, 1])
                with rev_col1:
                    st.image(selected_frame['img_path'], use_container_width=True)
                    
                with rev_col2:
                    st.markdown(f"**Timestamp:** {selected_frame['time']}")
                    st.markdown(f"**Classified Pose:** {selected_frame['pose']}")
                    st.markdown(f"**Confidence:** {selected_frame['conf']:.1f}%")
                    
                    if selected_frame['is_perfect']:
                        st.success("🟢 Perfect Posture!")
                        dynamic_bg_global.markdown("<style>.stApp { background-color: #4ade80 !important; }</style>", unsafe_allow_html=True)
                    else:
                        st.error("🔴 Feedback:")
                        for f in selected_frame['feedback']:
                            st.write(f"- {f}")
                        dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)
            else:
                # Display BIG SCORE if in Full Video Mode
                if st.session_state.overall_score is not None:
                    score = st.session_state.overall_score
                    color = "#27AE60" if score >= 85 else "#F39C12" if score >= 70 else "#E74C3C"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background-color: white; border-radius: 15px; border-left: 10px solid {color}; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); margin-bottom: 25px;">
                        <h3 style="margin:0; color: #7F8C8D;">Overall Session Score</h3>
                        <h1 style="margin:0; font-size: 3.5rem; color: {color};">{score:.1f} / 100</h1>
                        <p style="margin:0; color: #95A5A6; font-size: 14px;">(Score ignores initial setup time to evaluate your true held posture)</p>
                    </div>
                    """, unsafe_allow_html=True)

                # FULL VIDEO MODE: Gallery View instead of Scrubber
                st.markdown("### 📸 Key Moments Gallery")
                st.info("Showing only significant frames where your posture changed, improved, or degraded.")
                
                cols = st.columns(2)
                for i, frame_data in df.iterrows():
                    with cols[i % 2]:
                        st.markdown(f'<div style="background:white; padding:10px; border-radius:10px; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
                        st.image(frame_data['img_path'], use_container_width=True)
                        st.markdown(f"**{frame_data['time']} | {frame_data.get('event_label', '')}**")
                        st.markdown(f"Pose: {frame_data['pose']} | Conf: {frame_data['conf']:.1f}%")
                        if frame_data['is_perfect']:
                            st.success("🟢 Perfect Posture!")
                        else:
                            st.error("🔴 " + ", ".join(frame_data['feedback']))
                        st.markdown('</div>', unsafe_allow_html=True)
                        
            st.markdown("---")
            pdf_bytes = create_pdf_report(st.session_state.report_data, st.session_state.overall_score)
            st.download_button(
                label="📄 Download Complete PDF Report",
                data=pdf_bytes,
                file_name="Yoga_Video_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
    else:
        st.markdown("### Video File Analysis")
        st.info("Upload a video file from the sidebar to generate a statistical dashboard and report.")
        dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)

# --- 4. LEARNING Hub MODE ---
elif selected == "Learning Hub":
    dynamic_bg_global.markdown("<style>.stApp { background-color: #F4F7F6 !important; }</style>", unsafe_allow_html=True)
    st.markdown("### Interactive Asana Learning Hub")
    
    available_poses = list(yoga_ai.class_map.keys())
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_pose = st.selectbox("Select an Asana to study:", [pose.capitalize() for pose in available_poses])
    
    st.markdown("---")
    info_col, action_col = st.columns([1, 1])
    
    with info_col:
        st.markdown(f"### {selected_pose} Overview")
        st.markdown("""
        **AI Evaluation Metrics:**
        * **Spinal Alignment:** Ensuring core engagement and straight posture.
        * **Joint Extensions:** Checking specific Knee/Elbow angles against ideal tolerances.
        * **Balance & Symmetry:** Tracking weight distribution across the skeleton graph.
        """)
        
    with action_col:
        st.markdown("### Video Tutorials")
        search_query = selected_pose.lower().replace(' ', '+') + "+yoga+pose+step+by+step"
        youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
        
        st.link_button(f"Search YouTube for '{selected_pose}' Tutorials", youtube_url, use_container_width=True)
