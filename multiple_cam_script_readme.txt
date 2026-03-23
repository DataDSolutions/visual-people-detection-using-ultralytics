Multi-camera people detection and tracking system using YOLO for object detection and ResNet18-based ReID (Re-Identification) for cross-camera person tracking.

Custom UID system assigns unique identifiers to people across cameras using appearance embeddings (cosine similarity matching with 0.35 threshold).

Processes multiple RTSP streams (from comma-separated URLs or text file), resizes frames to 640x480, runs YOLO detection with configurable confidence threshold.

Categorizes detections into Female/Male/Child Boy/Child Girl using a predefined mapping of YOLO class labels (handles sitting/standing variants).

Real-time visualization displays all camera feeds in a responsive grid with bounding boxes, confidence scores, UID labels, and live people counts per category.

Generates daily CSV logs (one per camera) in logs/YYYY-MM-DD/cam_X.csv format tracking timestamped counts of each category plus total detections.

Active tracking management tracks people appearing/disappearing per camera, calculates dwell time per camera per person, auto-cleans old tracks after 2 hours.

Performance monitoring maintains rolling FPS average (200-frame buffer) and handles camera failures gracefully with black placeholder frames.

Clean shutdown properly releases video captures, closes log files, and destroys OpenCV windows when 'q' is pressed.