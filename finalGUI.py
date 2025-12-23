import sys
import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QCheckBox, QFrame, QSlider, QStyle)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"
IMAGE_FOLDER = "Fire_Evidence_Images"
VIDEO_FOLDER = "Fire_Evidence_Videos"

if not os.path.exists(IMAGE_FOLDER): os.makedirs(IMAGE_FOLDER)
if not os.path.exists(VIDEO_FOLDER): os.makedirs(VIDEO_FOLDER)

class FireDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FireGuard: Pro Dashboard")
        self.setGeometry(100, 100, 1280, 850)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # --- LOGIC VARIABLES ---
        self.model = YOLO(MODEL_PATH)
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Playback State
        self.is_running = False
        self.is_paused = False
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        
        # Toggles
        self.show_debug_zones = True
        self.enforce_safe_zone = True
        
        # Recording logic
        self.recording = False
        self.video_writer = None
        self.patience_counter = 0
        self.PATIENCE_LIMIT = 30
        
        # --- GUI LAYOUT ---
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout() 
        main_widget.setLayout(main_layout)

        # ================= LEFT SIDE: VIDEO & PLAYBACK =================
        left_layout = QVBoxLayout()
        
        # 1. Video Screen
        self.video_label = QLabel("No Video Source Loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #444; background-color: black; font-size: 16px;")
        self.video_label.setMinimumSize(800, 500)
        left_layout.addWidget(self.video_label, stretch=1)

        # 2. Playback Controls
        playback_panel = QFrame()
        playback_panel.setStyleSheet("background-color: #252525; border-radius: 8px; padding: 5px;")
        playback_layout = QHBoxLayout()
        playback_panel.setLayout(playback_layout)

        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setFixedSize(40, 40)
        self.btn_play.setStyleSheet("background-color: #0078d7; border-radius: 20px;")
        self.btn_play.clicked.connect(self.toggle_play_pause)
        playback_layout.addWidget(self.btn_play)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: monospace; font-size: 12px; color: #aaa;")
        playback_layout.addWidget(self.lbl_time)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { background: #0078d7; width: 16px; margin: -5px 0; border-radius: 8px; }
        """)
        playback_layout.addWidget(self.slider)

        left_layout.addWidget(playback_panel)
        main_layout.addLayout(left_layout, stretch=3)

        # ================= RIGHT SIDE: CONTROL PANEL =================
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; padding: 15px;")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        # Title
        title = QLabel("FIRE GUARD")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ff5500; letter-spacing: 2px;")
        control_layout.addWidget(title)
        control_layout.addSpacing(20)

        # Import Buttons
        btn_style = """
            QPushButton { background-color: #3a3a3a; color: white; font-size: 14px; padding: 12px; border-radius: 6px; border: 1px solid #555; }
            QPushButton:hover { background-color: #4a4a4a; border: 1px solid #777; }
        """
        self.btn_load_video = QPushButton("📂 Import Video File")
        self.btn_load_video.setStyleSheet(btn_style)
        self.btn_load_video.clicked.connect(self.load_video)
        control_layout.addWidget(self.btn_load_video)
        
        self.btn_load_cam = QPushButton("📷 Use Webcam (Live)")
        self.btn_load_cam.setStyleSheet(btn_style)
        self.btn_load_cam.clicked.connect(self.load_webcam)
        control_layout.addWidget(self.btn_load_cam)
        
        control_layout.addSpacing(20)
        
        # Settings
        lbl_settings = QLabel("DETECTION SETTINGS")
        lbl_settings.setStyleSheet("font-size: 12px; font-weight: bold; color: #888;")
        control_layout.addWidget(lbl_settings)

        self.check_rule = QCheckBox("Enforce Safe Area (Strict)")
        self.check_rule.setChecked(True)
        self.check_rule.stateChanged.connect(self.toggle_rule_logic)
        self.check_rule.setStyleSheet("font-size: 14px; color: #ffcc00; margin-top: 5px;")
        control_layout.addWidget(self.check_rule)
        
        self.check_zones = QCheckBox("Show Debug Zones")
        self.check_zones.setChecked(True)
        self.check_zones.stateChanged.connect(self.toggle_zones)
        self.check_zones.setStyleSheet("font-size: 14px; margin-top: 5px;")
        control_layout.addWidget(self.check_zones)

        control_layout.addSpacing(40)

        # Live Stats
        stats_box = QFrame()
        stats_box.setStyleSheet("background-color: #1a1a1a; border-radius: 8px; padding: 10px;")
        stats_layout = QVBoxLayout()
        stats_box.setLayout(stats_layout)
        
        self.status_label = QLabel("IDLE")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #555;")
        stats_layout.addWidget(self.status_label)
        
        self.stats_label = QLabel("Fire: 0  |  Smoke: 0")
        self.stats_label.setFont(QFont("Arial", 14))
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("color: #ccc; margin-top: 5px;")
        stats_layout.addWidget(self.stats_label)
        
        control_layout.addWidget(stats_box)
        control_layout.addStretch()

        self.btn_stop = QPushButton("Stop Process")
        self.btn_stop.setStyleSheet("background-color: #d70000; color: white; padding: 12px; border-radius: 6px; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_process)
        control_layout.addWidget(self.btn_stop)

        main_layout.addWidget(control_panel, stretch=1)

    # --- VIDEO HANDLING ---
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.start_stream(file_path)

    def load_webcam(self):
        self.start_stream(0)

    def start_stream(self, source):
        self.stop_process()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.video_label.setText("Error: Could not open source")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        if self.total_frames > 0:
            self.slider.setRange(0, self.total_frames)
            self.slider.setEnabled(True)
        else:
            self.slider.setEnabled(False)

        self.is_running = True
        self.is_paused = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.timer.start(int(1000/self.fps))

    def stop_process(self):
        self.is_running = False
        self.timer.stop()
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()
        
        self.recording = False
        self.video_writer = None
        self.video_label.setText("System Idle")
        self.video_label.clear()
        self.status_label.setText("IDLE")
        self.status_label.setStyleSheet("color: #555;")
        self.slider.setValue(0)
        self.lbl_time.setText("00:00 / 00:00")

    def toggle_play_pause(self):
        if not self.is_running: return
        if self.is_paused:
            self.is_paused = False
            self.timer.start()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.is_paused = True
            self.timer.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def slider_pressed(self):
        self.was_playing = not self.is_paused
        if self.was_playing:
            self.timer.stop()
            self.is_paused = True

    def slider_released(self):
        frame_idx = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if self.was_playing:
            self.timer.start()
            self.is_paused = False
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def slider_moved(self, position):
        self.update_time_label(position)

    def update_time_label(self, frame_idx):
        if self.fps <= 0: return
        seconds = int(frame_idx / self.fps)
        total_seconds = int(self.total_frames / self.fps) if self.total_frames > 0 else 0
        self.lbl_time.setText(f"{seconds//60:02}:{seconds%60:02} / {total_seconds//60:02}:{total_seconds%60:02}")

    def toggle_zones(self): self.show_debug_zones = self.check_zones.isChecked()
    def toggle_rule_logic(self): self.enforce_safe_zone = self.check_rule.isChecked()

    def update_frame(self):
        if not self.is_running: return
        ret, frame = self.cap.read()
        if not ret:
            self.is_paused = True
            self.timer.stop()
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            return

        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.slider.blockSignals(True)
        self.slider.setValue(current_pos)
        self.slider.blockSignals(False)
        self.update_time_label(current_pos)

        processed_frame, fire_c, smoke_c, status = self.process_frame(frame)
        
        self.stats_label.setText(f"Fire: {fire_c}  |  Smoke: {smoke_c}")
        if status == "CRITICAL":
            self.status_label.setText("CRITICAL")
            self.status_label.setStyleSheet("color: #ff3333; font-weight: bold;")
        elif status == "WARNING":
            self.status_label.setText("WARNING")
            self.status_label.setStyleSheet("color: #ffcc00; font-weight: bold;")
        else:
            self.status_label.setText("SAFE")
            self.status_label.setStyleSheet("color: #00dd00; font-weight: bold;")

        # CV2 BGR -> Qt RGB
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def process_frame(self, frame):
        results = self.model(frame, stream=True, conf=0.09, imgsz=1280)
        fire_count, smoke_count = 0, 0
        temp_fire, temp_smoke = [], []

        for r in results:
            for box in r.boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy().astype(int)
                if cls == 0: temp_fire.append({'coords': coords, 'conf': conf})
                elif cls == 1: temp_smoke.append({'coords': coords, 'conf': conf})

        central_zones = []
        for smoke in temp_smoke:
            sx1, sy1, sx2, sy2 = smoke['coords']
            smoke_count += 1
            w, h = sx2 - sx1, sy2 - sy1
            pad_w, pad_h = int(w*0.125), int(h*0.125)
            central_zones.append([sx1+pad_w, sy1+pad_h, sx2-pad_w, sy2-pad_h])
            
            # SMOKE = GREY (128, 128, 128)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (128, 128, 128), 2)
            cvzone.putTextRect(frame, f"SMOKE {int(smoke['conf']*100)}%", 
                               (max(0, sx1), max(35, sy1-10)), 
                               scale=1, thickness=1, 
                               colorR=(128, 128, 128), # Background Grey
                               colorT=(255, 255, 255)) # Text White

            if self.show_debug_zones:
                # DEBUG ZONE = BLUE (255, 0, 0) in BGR
                cv2.rectangle(frame, (sx1+pad_w, sy1+pad_h), (sx2-pad_w, sy2-pad_h), (255, 0, 0), 1)

        for fire in temp_fire:
            fx1, fy1, fx2, fy2 = fire['coords']
            fw, fh = fx2 - fx1, fy2 - fy1
            
            if self.enforce_safe_zone:
                if not central_zones: continue
                if not self.is_in_central_smoke_area([fx1, fy1, fx2, fy2], central_zones): continue
            
            rx1, ry1, rx2, ry2 = self.refine_fire_box(frame, [fx1, fy1, fx2, fy2])
            rw, rh = rx2 - rx1, ry2 - ry1
            
            final_box = [rx1, ry1, rx2, ry2] if (rw*rh) < (fw*fh * 0.95) and rw > 5 and rh > 5 else [fx1, fy1, fx2, fy2]
            x1, y1, x2, y2 = final_box
            
            if (x2-x1) > 5 and (y2-y1) > 5:
                fire_count += 1
                # FIRE = RED (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f"FIRE {int(fire['conf']*100)}%", 
                                   (max(0, x1), max(35, y1-10)), 
                                   scale=1, thickness=1, 
                                   colorR=(0, 0, 255), # Background Red
                                   colorT=(255, 255, 255)) # Text White

        status = "CRITICAL" if fire_count > 0 else "WARNING" if smoke_count > 0 else "SAFE"
        self.handle_recording(frame, fire_count, smoke_count)
        return frame, fire_count, smoke_count, status

    def handle_recording(self, frame, fire_c, smoke_c):
        if fire_c > 0 or smoke_c > 0:
            self.patience_counter = self.PATIENCE_LIMIT
            if not self.recording:
                self.recording = True
                ts = int(time.time())
                self.video_writer = cv2.VideoWriter(f"{VIDEO_FOLDER}/Event_{ts}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame.shape[1], frame.shape[0]))
                cv2.imwrite(f"{IMAGE_FOLDER}/Capture_{ts}.jpg", frame)
                print(f"[REC] Started: Event_{ts}.mp4")
        elif self.recording:
            self.patience_counter -= 1
            if self.patience_counter <= 0:
                self.recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    print("[REC] Stopped")
        
        if self.recording and self.video_writer:
            self.video_writer.write(frame)
            cv2.circle(frame, (frame.shape[1]-30, 30), 10, (0, 0, 255), -1)

    def refine_fire_box(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if roi.size == 0: return box
        mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), np.array([0, 120, 180]), np.array([25, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            bx, by, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
            if bw > 5 and bh > 5 and bw < (x2 - x1): return x1+bx, y1+by, x1+bx+bw, y1+by+bh
        return box

    def is_in_central_smoke_area(self, fire_box, zones):
        f_cx, f_cy = (fire_box[0] + fire_box[2]) // 2, (fire_box[1] + fire_box[3]) // 2
        for z in zones:
            if z[0] < f_cx < z[2] and z[1] < f_cy < z[3]: return True
        return False

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec_())