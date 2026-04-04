import cv2
import yaml
from detector import VehicleDetector

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {2: (0,255,0), 3: (255,165,0), 5: (255,0,0), 7: (0,165,255)}

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

detector = VehicleDetector()

source = cfg["video"]["source"]
cap = cv2.VideoCapture(source)

print("Running detection — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    # Draw each detection
    for x1, y1, x2, y2, conf, cls in detections:
        color = COLORS.get(cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls, 'vehicle')} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Show count
    cv2.putText(frame, f"Vehicles detected: {len(detections)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Resize for display
    display = cv2.resize(frame, (900, int(frame.shape[0] * 900 / frame.shape[1])))
    cv2.imshow("Phase 2 - Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")