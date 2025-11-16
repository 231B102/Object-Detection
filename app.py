import cv2
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import time
import threading

app = Flask(__name__)

# Load the model
model = YOLO("yolov8m.pt")

current_detections = []
# A lock is good practice to prevent "read/write" conflicts between the camera and the API
lock = threading.Lock() 

def generate_frames():
    global current_detections
    
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, stream=True, verbose=False)

        frame_detections = []
        height, width, _ = frame.shape
        
        for r in results:
            annotated_frame = r.plot()
            
            for box in r.boxes:
                # --- NEW: CONFIDENCE FILTER ---
                confidence = float(box.conf[0])
                
                # If confidence is less than 0.75 (75%), skip this object
                if confidence < 0.75:
                    continue
                # ------------------------------

                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                
                position = "center"
                if center_x < width / 3:
                    position = "left"
                elif center_x > (width / 3) * 2:
                    position = "right"
                
                frame_detections.append({
                    "label": class_name,
                    "position": position,
                    "confidence": round(confidence, 2)
                })
            
            frame = annotated_frame

        # Update the global variable safely using the lock
        with lock:
            current_detections = frame_detections

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """API endpoint for the frontend to fetch current objects."""
    with lock:
        return jsonify(current_detections)

if __name__ == '__main__':
    print("Starting Accessibility Helper...")
    print("Please open http://127.0.0.1:5500 in your browser")
    app.run(debug=True, threaded=True, port=5500)