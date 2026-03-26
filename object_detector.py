# object_detector.py
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_objects(frame, target_class=None, conf=0.4):
    results = model(frame, conf=conf, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        if target_class is None or cls_name == target_class:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append({
                "class": cls_name,
                "conf": float(box.conf),
                "center": ((x1+x2)//2, (y1+y2)//2),
                "box": (x1,y1,x2,y2)
            })
    return detections

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()   
    if not ret:
        break
    
    found, center = detect_objects(frame, 'red')
    
    cv2.imshow("Color Detection - Press ESC to Exit", frame)
    
    if cv2.waitKey(1) == 27:
        break 

cap.release()
cv2.destroyAllWindows()