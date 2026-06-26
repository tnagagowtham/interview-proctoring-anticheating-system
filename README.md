# 🎥 Real-Time Interview Proctoring & Anti-Cheating System

A real-time monitoring system for online interviews that detects suspicious
behavior and enforces strict anti-cheating controls.

## 🔗 Live Demo
**[Try the live detection demo](https://huggingface.co/spaces/tnagagowtham/interview-proctoring-demo)** — face + object detection running live in your browser.

⚠️ **Note:** The live demo runs on free CPU hardware, so detection is slower
than real-time. The full system runs smoothly in real time locally with GPU
support — see the demo video below for full-speed performance.

## 🚀 Overview
Tracks candidate head movement, face presence, and object detection to flag
suspicious behavior during online interviews. Uses a 3-strike warning system
that auto-terminates the session on the 4th violation, with full-screen and
keyboard lock to prevent tab-switching and unauthorized input.

## 🛠️ Tech Stack
Python, YOLO, Face Recognition, HTML, CSS, JavaScript

## ✨ Key Features
- Tracks head movement and face presence in real time
- Object detection to flag unauthorized items (e.g. phone, notes) in frame
- **3-strike warning system** — visible warnings on the first 3 violations
- **Auto-termination** of the interview on the 4th violation
- Full-screen lock to prevent minimizing or switching windows
- Keyboard lock to block unauthorized typing/shortcuts during the session

## 🔌 How It Works
1. Candidate's webcam feed is analyzed frame-by-frame using YOLO + face recognition.
2. System checks: is a face present? Is the head facing the screen? Are there
   unauthorized objects in frame?
3. Each violation triggers a strike (1st–3rd = warning).
4. On the 4th violation, the interview session is automatically terminated.
5. Full-screen and keyboard locks run throughout to block tab-switching
   and unauthorized input.

## 🎬 Demo Video
*(Drag your screen recording directly into this README's editor on GitHub —
it will embed and play inline here, showing the full lockdown behavior that
the live web demo above cannot replicate.)*

## 📈 Future Improvements
- Audio analysis to detect whispering/second voices
- Admin dashboard with violation logs and timestamped clips
- Configurable strike thresholds per interview type

## 📌 Status
Core detection, strike system, and screen/keyboard lock implemented and tested.
