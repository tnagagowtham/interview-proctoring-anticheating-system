# professional_interview_simple.py
import cv2
import numpy as np
import time
from datetime import datetime
import os
import json
from collections import deque
import threading
from flask import Flask, render_template_string, jsonify, Response, request
import webbrowser
import atexit

class SimpleInterviewMonitor:
    def __init__(self):
        print("\n" + "="*50)
        print("PROFESSIONAL INTERVIEW MONITOR")
        print("="*50)
        
        self.records_dir = "records"
        os.makedirs(self.records_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = self.initialize_camera()
        if not self.cap:
            print("[ERROR] Camera initialization failed")
            return
        
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load YOLO model for object detection
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            print("[OK] YOLOv8 model loaded")
        except Exception as e:
            print(f"[!] Failed to load YOLOv8 model: {e}")
            self.yolo_model = None

        self.suspicious_objects = ['cell phone', 'book', 'laptop', 'tablet']
        
        # Interview state
        self.candidate_name = None
        self.interview_active = False
        self.start_time = None
        self.deviations = []
        self.video_writer = None
        
        # Tracking
        self.current_status = "READY"
        self.face_position_history = deque(maxlen=20)
        
        # Flask server
        self.app = Flask(__name__)
        self.setup_routes()
        
        print("\n[OK] System ready")
        print("Starting web interface...")
        self.start_web_interface()
    
    def initialize_camera(self):
        """Initialize camera with compatibility"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            # Try different indices
            for i in range(4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"[OK] Camera found at index {i}")
                    break
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("[OK] Camera configured")
        else:
            print("[!] Using simulated camera mode")
        
        return cap
    
    def generate_video_feed(self):
        """Generate video feed with face tracking and YOLO"""
        while True:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = self.process_frame(frame)
                    
                    if self.interview_active and self.video_writer is not None:
                        try:
                            self.video_writer.write(frame)
                        except Exception as e:
                            print(f"[!] Frame write error: {e}")
                else:
                    # Create test frame if camera fails
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA NOT AVAILABLE", (50, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Create test frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "INTERVIEW MONITOR", (100, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Face Tracking Active", (120, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def process_frame(self, frame):
        """Process frame with face tracking and YOLO"""
        height, width = frame.shape[:2]
        display = frame.copy()
        
        # Add subtle center guide
        center_x, center_y = width // 2, height // 2
        cv2.line(display, (center_x, center_y - 15), (center_x, center_y + 15),
                (100, 100, 100), 1)
        cv2.line(display, (center_x - 15, center_y), (center_x + 15, center_y),
                (100, 100, 100), 1)
        
        status = "READY"
        color = (50, 220, 50)  # Green
        thickness = 2
        deviation_msg = None

        # 1. Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            status = "ALERT"
            color = (50, 50, 220)  # Red
            deviation_msg = "Face not detected in frame"
        else:
            # We assume the largest face is the candidate
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            distance_from_center = np.sqrt(
                (face_center_x - center_x)**2 + 
                (face_center_y - center_y)**2
            )
            
            if distance_from_center > 120:
                status = "ALERT"
                color = (50, 50, 220)
                deviation_msg = "Position deviation (out of bounds)"
            elif distance_from_center > 60:
                status = "WARNING"
                color = (50, 180, 220)
            
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            cv2.circle(display, (face_center_x, face_center_y), 4, (255, 255, 255), -1)
            if distance_from_center > 30:
                cv2.line(display, (face_center_x, face_center_y),
                        (center_x, center_y), color, 1)

        # 2. YOLO Object Detection
        if self.yolo_model:
            y_res = self.yolo_model(frame, verbose=False, conf=0.3)[0]
            person_count = 0
            for box in y_res.boxes:
                cls_id = int(box.cls[0])
                label = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                
                if label == 'person':
                    person_count += 1
                elif label in self.suspicious_objects and conf > 0.4:
                    status = "ALERT"
                    color = (50, 50, 220)
                    deviation_msg = "OBJECT DETECTED"
                    # Draw box around suspicious object
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (50, 50, 220), 3)
                    cv2.putText(display, "OBJECT DETECTED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 220), 2)
            
            if person_count > 1:
                status = "ALERT"
                color = (50, 50, 220)
                deviation_msg = "MULTIPLE PEOPLE DETECTED"

        self.current_status = status
        if status == "ALERT" and self.interview_active and deviation_msg:
            self.record_deviation(deviation_msg)
        
        # Add status indicator
        status_color = {
            "READY": (50, 220, 50),
            "WARNING": (50, 180, 220),
            "ALERT": (50, 50, 220)
        }.get(self.current_status, (100, 100, 100))
        
        cv2.circle(display, (width - 30, 30), 8, status_color, -1)
        
        if self.candidate_name:
            cv2.putText(display, self.candidate_name, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        
        if self.interview_active and self.start_time:
            elapsed = int(time.time() - self.start_time)
            timer_text = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            cv2.putText(display, timer_text, (width - 100, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        
        return display
    
    def record_deviation(self, message):
        """Record position deviation"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        deviation = {
            "time": timestamp,
            "message": message,
            "status": self.current_status
        }
        
        if not self.deviations or self.deviations[-1]["message"] != message:
            self.deviations.append(deviation)
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(SIMPLE_UI)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_video_feed(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @self.app.route('/status')
        def get_status():
            status = {
                "candidate": self.candidate_name,
                "status": self.current_status,
                "interview_active": self.interview_active,
                "deviations_count": len(self.deviations),
                "recent_deviations": self.deviations[-5:] if self.deviations else []
            }
            return jsonify(status)
        
        @self.app.route('/start_interview', methods=['POST'])
        def start_interview():
            data = request.get_json()
            self.candidate_name = data.get('name', 'Candidate')
            self.interview_active = True
            self.start_time = time.time()
            self.deviations = []
            
            if not self.cap or not self.cap.isOpened():
                self.cap = self.initialize_camera()
            
            if self.cap and self.cap.isOpened():
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width == 0 or height == 0:
                    width, height = 1280, 720
                fps = 15
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.records_dir, f"{self.candidate_name.replace(' ', '_')}_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                print(f"[OK] Started recording to {filename}")
                
            return jsonify({"success": True})
        
        @self.app.route('/end_interview', methods=['POST'])
        def end_interview():
            self.interview_active = False
            
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print("[OK] Video recording stopped and saved.")
                
            if self.cap:
                self.cap.release()
                self.cap = None
                print("[OK] Camera released.")
            
            report = self.generate_report()
            return jsonify({"success": True, "report": report})
    
    def start_web_interface(self):
        """Start web interface in background"""
        threading.Thread(
            target=lambda: self.app.run(
                host='0.0.0.0',
                port=5000,
                debug=False,
                use_reloader=False
            ),
            daemon=True
        ).start()
        
        # Open browser after delay
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    def generate_report(self):
        """Generate interview report"""
        if not self.start_time:
            return "No interview data"
        
        duration = time.time() - self.start_time
        alert_count = sum(1 for d in self.deviations if d["status"] == "ALERT")
        
        report = f"""
Interview Report
===============
Candidate: {self.candidate_name}
Duration: {duration:.0f} seconds
Deviations: {len(self.deviations)}
Significant alerts: {alert_count}

Summary:
- {'Good focus maintained' if alert_count < 3 else 'Focus improvements needed'}
"""
        return report
    
    def run(self):
        """Main loop"""
        print("\n[INFO] Professional Interview Monitor Running")
        print("Open http://localhost:5000 in your browser")
        print("Press Ctrl+C to exit\n")
        
        try:
            # Keep program running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")
            if self.cap:
                self.cap.release()

SIMPLE_UI = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Interview Portal</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            user-select: none;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #000000 0%, #111111 100%);
            color: #FFD700;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            padding: 20px 40px;
            background: linear-gradient(90deg, rgba(20,20,20,0.95), rgba(40,35,0,0.9));
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #DAA520;
            box-shadow: 0 4px 15px rgba(218, 165, 32, 0.2);
            z-index: 10;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 300;
            letter-spacing: 2px;
            color: #FFD700;
            text-transform: uppercase;
        }
        
        .header-status {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .status-badge {
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            border: 1px solid #FFD700;
            background: rgba(218, 165, 32, 0.1);
            color: #FFD700;
            transition: all 0.3s ease;
        }
        
        .main-content {
            display: flex;
            flex: 1;
            padding: 20px;
            gap: 20px;
            height: calc(100vh - 80px);
        }
        
        .video-container {
            flex: 2;
            background: #050505;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            border: 2px solid #332a00;
            box-shadow: 0 0 30px rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .video-feed {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .video-overlay {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px 20px;
            border-radius: 8px;
            border-left: 4px solid #FFD700;
            backdrop-filter: blur(5px);
        }
        
        .side-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .panel {
            background: rgba(20,20,20,0.8);
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #332a00;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        }
        
        .panel h2 {
            font-size: 16px;
            margin-bottom: 20px;
            color: #DAA520;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 1px solid #332a00;
            padding-bottom: 10px;
        }
        
        .logs-container {
            flex: 1;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 13px;
        }
        
        .log-entry {
            margin-bottom: 12px;
            padding: 10px;
            background: rgba(0,0,0,0.4);
            border-left: 3px solid #DAA520;
            color: #eee;
        }
        
        .log-entry.alert {
            border-left-color: #ff3333;
            color: #ffaaaa;
            background: rgba(255, 0, 0, 0.1);
        }
        
        .btn {
            padding: 15px 30px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s;
            width: 100%;
        }
        
        .btn-end {
            background: linear-gradient(135deg, #cc0000, #880000);
            color: white;
            border: 1px solid #ff4444;
            box-shadow: 0 4px 15px rgba(204, 0, 0, 0.3);
        }
        
        .btn-end:hover {
            background: linear-gradient(135deg, #ff0000, #cc0000);
            box-shadow: 0 4px 20px rgba(255, 0, 0, 0.5);
            transform: translateY(-2px);
        }

        .btn-start {
            background: linear-gradient(135deg, #FFD700, #DAA520);
            color: #000;
            border: none;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(218, 165, 32, 0.4);
        }

        .btn-start:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 25px rgba(218, 165, 32, 0.6);
        }
        
        /* Modals and Overlays */
        .candidate-form-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.95);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            backdrop-filter: blur(10px);
        }

        .candidate-form-box {
            background: #111;
            padding: 50px;
            border-radius: 12px;
            border: 2px solid #DAA520;
            text-align: center;
            box-shadow: 0 0 40px rgba(218, 165, 32, 0.2);
            width: 450px;
        }

        .candidate-form-box input {
            width: 100%;
            padding: 15px;
            margin-top: 20px;
            background: #222;
            border: 1px solid #DAA520;
            color: #FFD700;
            font-size: 18px;
            border-radius: 6px;
            outline: none;
        }

        /* Alert Popup */
        .alert-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 3000;
            backdrop-filter: blur(8px);
        }

        .alert-box {
            background: #111;
            padding: 60px;
            border-radius: 15px;
            border: 4px solid #fff;
            text-align: center;
            box-shadow: 0 0 50px rgba(255, 255, 255, 0.4);
            color: white;
            max-width: 600px;
        }

        .alert-box h1 {
            font-size: 50px;
            margin-bottom: 20px;
            color: #ff3333;
            text-transform: uppercase;
        }

        .alert-box p {
            font-size: 24px;
            margin-bottom: 40px;
        }

        .alert-box button {
            padding: 20px 40px;
            font-size: 22px;
            background: #ff3333;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            text-transform: uppercase;
            font-weight: bold;
            transition: transform 0.2s;
        }
        
        .alert-box button:hover {
            transform: scale(1.05);
            background: #cc0000;
        }

        /* Tab Switch / Exited Fullscreen Warning */
        .security-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.98);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2500;
            flex-direction: column;
            color: #FFD700;
            text-align: center;
        }
        .security-overlay h1 { font-size: 40px; color: #ff3333; margin-bottom: 20px; }
        .security-overlay p { font-size: 20px; margin-bottom: 40px; color: #DAA520; }
        .security-overlay button { width: auto; padding: 15px 40px; }

        /* Notification Toast */
        #notification {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(218, 165, 32, 0.9);
            color: #000;
            padding: 15px 30px;
            border-radius: 30px;
            font-weight: bold;
            display: none;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body>
    
    <!-- Initial Candidate Form -->
    <div class="candidate-form-overlay" id="candidateForm">
        <div class="candidate-form-box">
            <h1 style="color: #FFD700; margin-bottom: 10px;">ENTERPRISE PORTAL</h1>
            <p style="color: #aaa; margin-bottom: 20px;">Please enter your full name to begin</p>
            <input type="text" id="candidateName" placeholder="Your Name" autocomplete="off">
            <button class="btn btn-start" onclick="submitCandidate()">Start Secure Interview</button>
        </div>
    </div>

    <!-- Massive Red Alert Box (For Object/Face Violations) -->
    <div class="alert-overlay" id="alertOverlay">
        <div class="alert-box">
            <h1>⚠️ SECURITY ALERT</h1>
            <p id="alertReason">Unauthorized activity detected.</p>
            <button onclick="dismissAlert()">ACKNOWLEDGE & RESUME</button>
        </div>
    </div>

    <!-- Security Overlay (For Tab Switch / Esc) -->
    <div class="security-overlay" id="securityOverlay">
        <h1>⚠️ SECURITY VIOLATION</h1>
        <p>You exited fullscreen or switched tabs.<br>
           <span style="font-size: 16px; color: #888;">(Note: System keys like Windows/Esc cannot be blocked by browsers, but using them is recorded as a violation.)</span>
        </p>
        <button class="btn btn-start" onclick="resumeInterview()">Acknowledge & Resume Fullscreen</button>
    </div>

    <!-- Notification Toast -->
    <div id="notification">Notification message</div>

    <div class="header">
        <h1>Enterprise Interview Portal</h1>
        <div class="header-status">
            <span id="timer" style="font-family: Consolas; font-size: 18px; color: #aaa;">00:00</span>
            <div id="statusBadge" class="status-badge">READY</div>
        </div>
    </div>
    
    <div class="main-content">
        <div class="video-container">
            <img src="/video_feed" class="video-feed" alt="Video Feed">
            <div class="video-overlay">
                <div style="font-size: 12px; color: #DAA520; margin-bottom: 4px;">SECURE STREAM</div>
                <div id="candidateDisplay" style="font-weight: bold; font-size: 16px;">Candidate</div>
            </div>
        </div>
        
        <div class="side-panel">
            <div class="panel" style="flex: 1; display: flex; flex-direction: column;">
                <h2>Security Logs</h2>
                <div class="logs-container" id="logsContainer">
                    <div class="log-entry" style="color: #aaa; border-color: #555;">
                        System initialized. Waiting for candidate.
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <button class="btn btn-end" onclick="endInterview()">End Interview</button>
            </div>
        </div>
    </div>
    
    <script>
        let interviewActive = false;
        let alertPopupOpen = false;
        
        function showNotification(msg) {
            const notif = document.getElementById('notification');
            notif.innerText = msg;
            notif.style.display = 'block';
            setTimeout(() => { notif.style.display = 'none'; }, 3000);
        }

        function updateStatus() {
            if (!interviewActive) return;
            
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const badge = document.getElementById('statusBadge');
                    badge.textContent = data.status;
                    
                    if (data.status === 'READY') {
                        badge.style.borderColor = '#00ff64';
                        badge.style.color = '#00ff64';
                        badge.style.background = 'rgba(0, 255, 100, 0.1)';
                    } else if (data.status === 'WARNING') {
                        badge.style.borderColor = '#FF8C00';
                        badge.style.color = '#FF8C00';
                        badge.style.background = 'rgba(255, 140, 0, 0.1)';
                    } else {
                        badge.style.borderColor = '#ff3333';
                        badge.style.color = '#ff3333';
                        badge.style.background = 'rgba(255, 51, 51, 0.1)';
                    }

                    // Open massive popup on ALERT
                    if (data.status === 'ALERT' && !alertPopupOpen) {
                        alertPopupOpen = true;
                        document.getElementById('alertOverlay').style.display = 'flex';
                        const reason = data.recent_deviations && data.recent_deviations.length > 0 
                            ? data.recent_deviations[data.recent_deviations.length - 1].message 
                            : "Security violation detected";
                        document.getElementById('alertReason').textContent = reason;
                    }
                    
                    if (data.candidate) {
                        document.getElementById('candidateDisplay').textContent = data.candidate;
                    }
                    
                    const logsHtml = data.recent_deviations.map(d => 
                        `<div class="log-entry ${d.status === 'ALERT' ? 'alert' : ''}">
                            <span style="color: #666">[${d.time}]</span> 
                            ${d.message}
                        </div>`
                    ).join('');
                    
                    if (logsHtml) {
                        document.getElementById('logsContainer').innerHTML = logsHtml;
                    }
                });
        }
        
        function dismissAlert() {
            alertPopupOpen = false;
            document.getElementById('alertOverlay').style.display = 'none';
        }

        function enterFullScreen() {
            let elem = document.documentElement;
            if (elem.requestFullscreen) {
                elem.requestFullscreen().catch(err => {
                    console.log("Error attempting to enable fullscreen:", err);
                });
            } else if (elem.webkitRequestFullscreen) {
                elem.webkitRequestFullscreen();
            } else if (elem.msRequestFullscreen) {
                elem.msRequestFullscreen();
            }
        }

        function submitCandidate() {
            const name = document.getElementById('candidateName').value.trim();
            if (name) {
                enterFullScreen();
                
                fetch('/start_interview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('candidateForm').style.display = 'none';
                        interviewActive = true;
                        updateStatus();
                    }
                });
            } else {
                alert("Please enter your name.");
            }
        }
        
        function endInterview() {
            if (confirm('End the interview session?')) {
                interviewActive = false;
                fetch('/end_interview', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Interview completed.\\n' + data.report);
                        location.reload();
                    }
                });
            }
        }
        
        function resumeInterview() {
            enterFullScreen();
            document.getElementById('securityOverlay').style.display = 'none';
        }

        // Initialize
        updateStatus();
        setInterval(updateStatus, 1000);
        
        // Timer
        let seconds = 0;
        setInterval(() => {
            if(interviewActive && !alertPopupOpen) {
                seconds++;
                let m = Math.floor(seconds / 60).toString().padStart(2, '0');
                let s = (seconds % 60).toString().padStart(2, '0');
                document.getElementById('timer').innerText = `${m}:${s}`;
            }
        }, 1000);

        // Security: AGGRESSIVE Keyboard Blocking
        function blockKeyboard(e) {
            if (interviewActive) {
                // Allow enter key only if candidate form is open (though interviewActive should be false then)
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                showNotification("Keyboard is strictly disabled during interview.");
                return false;
            } else {
                // Before interview starts, allow Enter to submit name
                if (e.type === 'keydown' && e.code === 'Enter' && document.getElementById('candidateForm').style.display !== 'none') {
                    submitCandidate();
                }
            }
        }

        window.addEventListener('keydown', blockKeyboard, { capture: true });
        window.addEventListener('keyup', blockKeyboard, { capture: true });
        window.addEventListener('keypress', blockKeyboard, { capture: true });
        document.addEventListener('keydown', blockKeyboard, { capture: true });
        document.addEventListener('keyup', blockKeyboard, { capture: true });
        document.addEventListener('keypress', blockKeyboard, { capture: true });

        // Prevent context menu (right click)
        document.addEventListener('contextmenu', event => event.preventDefault());

        // Security: Tab switch detection and Fullscreen exit detection
        document.addEventListener('visibilitychange', function() {
            if (interviewActive && document.visibilityState === 'hidden') {
                document.getElementById('securityOverlay').style.display = 'flex';
            }
        });

        document.addEventListener('fullscreenchange', function() {
            if (interviewActive && !document.fullscreenElement) {
                document.getElementById('securityOverlay').style.display = 'flex';
            }
        });
    </script>
</body>
</html>
'''

# Run the application
if __name__ == "__main__":
    # First, let's install required packages automatically
    try:
        import cv2
        import numpy as np
        import flask
        import ultralytics
        print("[OK] All dependencies are installed")
    except ImportError:
        print("\n[!] Some dependencies are missing")
        print("Installing required packages...")
        import subprocess
        import sys
        
        # Install compatible versions for Python 3.13
        packages = [
            "opencv-python",
            "numpy",
            "flask",
            "ultralytics",
            "keyboard"
        ]
        
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"[OK] Installed {package}")
            except:
                print(f"[!] Could not install {package}")
        
        print("\n[OK] Installation complete. Please run the script again.")
        sys.exit(1)
    
    # Create and run monitor
    monitor = SimpleInterviewMonitor()
    monitor.run()