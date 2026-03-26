import cv2
import numpy as np

COLOR_RANGES = {
    "red":    ([0, 120, 70],   [10, 255, 255]),
    "green":  ([36, 100, 100], [86, 255, 255]),
    "blue":   ([94, 80, 2],    [126, 255, 255]),
    "yellow": ([15, 100, 100], [35, 255, 255]),
}

def detect_and_draw(frame, color_name):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(COLOR_RANGES[color_name][0])
    upper = np.array(COLOR_RANGES[color_name][1])
    
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {color_name}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return True, (x + w//2, y + h//2)
            
    return False, None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()   
    if not ret:
        break
    
    found, center = detect_and_draw(frame, 'yellow')
    
    cv2.imshow("Color Detection - Press ESC to Exit", frame)
    
    if cv2.waitKey(1) == 27:
        break 

cap.release()
cv2.destroyAllWindows()