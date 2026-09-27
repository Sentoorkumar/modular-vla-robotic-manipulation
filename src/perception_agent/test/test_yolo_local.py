# test_yolo_local.py
from ultralytics import YOLO
import cv2

model = YOLO("../models/best.pt")

print("Model loaded successfully")
print(f"Number of classes: {len(model.names)}")
print("Sample classes:", list(model.names.values())[:10])

# Test on webcam - press Q to quit
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.45, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("YOLO Local Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()