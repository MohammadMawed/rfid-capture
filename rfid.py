#!/usr/bin/env python3
# coding: utf-8

import time
import cv2
import os
import numpy as np
from datetime import datetime
from pirc522 import RFID
from picamera2 import Picamera2
import threading
from queue import Queue
import pickle

# -------------------- CONFIGURATION --------------------
FACE_SIZE = (160, 160)  # Smaller size for faster processing
CAPTURE_WIDTH = 640  # Lower resolution for speed
CAPTURE_HEIGHT = 480
CONFIDENCE_THRESHOLD = 60  # Lower is more confident (0-100)
SHARPNESS_THRESHOLD = 30  # Lower threshold for easier capture
FACE_PADDING = 20  # Padding around detected face
FRAME_SKIP = 2  # Process every Nth frame
MIN_FACE_SIZE = (60, 60)  # Minimum face size to detect

# -------------------- CAMERA SETUP --------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT)},
    controls={"FrameRate": 15}  # Lower FPS for stability
)
picam2.configure(config)
picam2.start()
time.sleep(1)

# -------------------- FACE DETECTION SETUP --------------------
# Use DNN face detector - much faster and more accurate than Haar Cascade
modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"

# Check if DNN model exists, otherwise fall back to optimized Haar
use_dnn = os.path.exists(modelFile) and os.path.exists(configFile)

if use_dnn:
    print("Using DNN face detector (faster and more accurate)")
    face_net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
else:
    print("DNN model not found. Using optimized Haar Cascade")
    print("Download DNN model for better performance:")
    print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb")
    print("https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# -------------------- THREADING FOR CAMERA --------------------
frame_queue = Queue(maxsize=2)
latest_frame = None
frame_lock = threading.Lock()

def capture_frames():
    """Background thread to continuously capture frames"""
    global latest_frame
    while True:
        frame = picam2.capture_array()
        with frame_lock:
            latest_frame = frame

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# -------------------- OPTIMIZED UTILS --------------------
def get_frame():
    """Get latest frame from camera"""
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy()
    return None

def sharpness(frame):
    """Calculate sharpness using Laplacian - optimized version"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize for faster computation
    small = cv2.resize(gray, (160, 120))
    laplacian = cv2.Laplacian(small, cv2.CV_64F)
    return laplacian.var()

def detect_faces_dnn(frame):
    """Detect faces using DNN - much faster and more accurate"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Add padding
            x1 = max(0, x1 - FACE_PADDING)
            y1 = max(0, y1 - FACE_PADDING)
            x2 = min(w, x2 + FACE_PADDING)
            y2 = min(h, y2 + FACE_PADDING)
            faces.append((x1, y1, x2-x1, y2-y1))
    
    return faces

def detect_faces_haar(frame):
    """Detect faces using optimized Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Downsample for faster detection
    scale = 2
    small = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
    
    faces = face_cascade.detectMultiScale(
        small,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(MIN_FACE_SIZE[0]//scale, MIN_FACE_SIZE[1]//scale),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Scale back up
    return [(x*scale, y*scale, w*scale, h*scale) for (x, y, w, h) in faces]

def detect_faces(frame):
    """Detect faces using best available method"""
    if use_dnn:
        return detect_faces_dnn(frame)
    else:
        return detect_faces_haar(frame)

def preprocess_face(face_img):
    """Preprocess face for recognition"""
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    # Resize to standard size
    resized = cv2.resize(gray, FACE_SIZE)
    # Histogram equalization for better contrast
    equalized = cv2.equalizeHist(resized)
    return equalized

def draw_alignment_box(frame):
    """Draw alignment guide box"""
    h, w, _ = frame.shape
    box_w, box_h = int(w * 0.4), int(h * 0.5)
    x1, y1 = (w - box_w) // 2, (h - box_h) // 2
    x2, y2 = x1 + box_w, y1 + box_h
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return (x1, y1, x2, y2)

def is_face_centered(face_box, guide_box):
    """Check if face is reasonably centered in guide box"""
    fx, fy, fw, fh = face_box
    gx1, gy1, gx2, gy2 = guide_box
    
    face_center_x = fx + fw // 2
    face_center_y = fy + fh // 2
    guide_center_x = (gx1 + gx2) // 2
    guide_center_y = (gy1 + gy2) // 2
    
    # Allow 30% tolerance
    tolerance_x = (gx2 - gx1) * 0.3
    tolerance_y = (gy2 - gy1) * 0.3
    
    return (abs(face_center_x - guide_center_x) < tolerance_x and
            abs(face_center_y - guide_center_y) < tolerance_y)

# -------------------- WORKER REGISTRATION --------------------
def register_worker(uid_str):
    """Register new worker with face capture"""
    print(f"Registering new worker for UID: {uid_str}")
    
    folder = f"faces/{uid_str}"
    os.makedirs(folder, exist_ok=True)
    
    captured = 0
    target_images = 5  # Reduced from 5 to make it faster
    frame_count = 0
    last_capture_time = 0
    
    print("Please position your face in the green box")
    print(f"Need {target_images} clear images")
    
    while captured < target_images:
        frame = get_frame()
        if frame is None:
            continue
            
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Draw alignment guide
        guide_box = draw_alignment_box(frame)
        
        # Detect faces
        faces = detect_faces(frame)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Take first face
            
            # Check if face is centered
            if is_face_centered((x, y, w, h), guide_box):
                # Draw green rectangle for aligned face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Check sharpness
                face_roi = frame[y:y+h, x:x+w]
                if sharpness(face_roi) > SHARPNESS_THRESHOLD:
                    # Prevent too rapid captures
                    current_time = time.time()
                    if current_time - last_capture_time > 0.5:
                        # Save image
                        face_processed = preprocess_face(face_roi)
                        filename = f"{folder}/{captured}.jpg"
                        cv2.imwrite(filename, face_processed)
                        captured += 1
                        last_capture_time = current_time
                        print(f"Captured {captured}/{target_images}")
                        cv2.putText(frame, f"Captured: {captured}/{target_images}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Hold steady - image blurry", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Draw red rectangle for uncentered face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Center your face in the box", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show preview
        cv2.imshow("Register Worker", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to cancel
            print("Registration cancelled")
            cv2.destroyWindow("Register Worker")
            return False
    
    cv2.destroyWindow("Register Worker")
    print("Registration complete!")
    return True

# -------------------- TRAINING --------------------
def train_recognizer():
    """Train face recognizer with all registered faces"""
    if not os.path.exists("faces"):
        print("No faces directory found")
        return False
    
    labels, faces_data = [], []
    uid_map = {}
    label_id = 0
    
    print("Loading training data...")
    for uid in os.listdir("faces"):
        folder = f"faces/{uid}"
        if not os.path.isdir(folder):
            continue
        
        uid_map[label_id] = uid
        
        for img_file in os.listdir(folder):
            if not img_file.endswith('.jpg'):
                continue
            
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                faces_data.append(img)
                labels.append(label_id)
        
        label_id += 1
    
    if len(faces_data) == 0:
        print("No training data found")
        return False
    
    print(f"Training with {len(faces_data)} images from {len(uid_map)} workers...")
    face_recognizer.train(faces_data, np.array(labels))
    
    # Save the model and mapping
    face_recognizer.save("face_model.yml")
    with open("uid_map.pkl", "wb") as f:
        pickle.dump(uid_map, f)
    
    print("Face recognizer trained and saved!")
    return True

# -------------------- CLOCK IN/OUT --------------------
def clock_in(uid_str):
    """Clock in/out with face verification"""
    print(f"Clocking in for UID: {uid_str}")
    
    # Load model if not already loaded
    if not os.path.exists("face_model.yml") or not os.path.exists("uid_map.pkl"):
        print("No trained model found. Please train first.")
        return
    
    try:
        face_recognizer.read("face_model.yml")
        with open("uid_map.pkl", "rb") as f:
            uid_map = pickle.load(f)
    except:
        print("Error loading model")
        return
    
    # Reverse mapping to get label from UID
    label_from_uid = {uid: label for label, uid in uid_map.items()}
    
    if uid_str not in label_from_uid:
        print("This worker is not registered. Work time NOT recorded.")
        return
    
    expected_label = label_from_uid[uid_str]
    
    print("Please look at the camera...")
    frame_count = 0
    attempts = 0
    max_attempts = 50  # Try for ~5 seconds
    
    while attempts < max_attempts:
        frame = get_frame()
        if frame is None:
            continue
        
        frame_count += 1
        attempts += 1
        
        # Skip frames
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Detect faces
        faces = detect_faces(frame)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_processed = preprocess_face(face_roi)
            
            # Recognize face
            label, confidence = face_recognizer.predict(face_processed)
            
            # Check if it matches the RFID card
            if label == expected_label and confidence < CONFIDENCE_THRESHOLD:
                # Face matched!
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("images", exist_ok=True)
                filename = f"images/{uid_str}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                
                print(f"Face matched! Work time registered. Image saved as {filename}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Welcome! Confidence: {confidence:.1f}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Clock In", frame)
                cv2.waitKey(2000)
                cv2.destroyWindow("Clock In")
                return
            else:
                # Face doesn't match
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Face does not match card", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Clock In", frame)
        cv2.waitKey(1)
    
    print("Face does not match registered worker. Work time NOT registered.")
    cv2.destroyWindow("Clock In")

# -------------------- MAIN --------------------
def main():
    rdr = RFID(pin_irq=None, pin_rst=22, pin_ce=0)
    
    # Create directories
    os.makedirs("faces", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    print("\n" + "="*50)
    print("OPTIMIZED RFID FACE RECOGNITION SYSTEM")
    print("="*50)
    print("1. Register new worker")
    print("2. Clock in / Work time")
    print("="*50)
    choice = input("Enter 1 or 2: ")
    
    try:
        while True:
            # Read RFID
            (error, tag_type) = rdr.request()
            if not error:
                (error, uid) = rdr.anticoll()
                if not error:
                    uid_str = "{:02X}{:02X}{:02X}{:02X}".format(*uid)
                    print(f"\nCard Detected! UID: {uid_str}")
                    
                    if choice == "1":
                        if register_worker(uid_str):
                            train_recognizer()
                    elif choice == "2":
                        clock_in(uid_str)
                    
                    time.sleep(1)  # Prevent double reads
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rdr.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
