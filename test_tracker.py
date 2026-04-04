import cv2
import yaml
from detector import VehicleDetector
from tracker import VehicleTracker

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Assign a unique color per track_id so you can visually follow each vehicle
def get_color(track_id):
    import hashlib
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    r = (h & 0xFF0000) >> 16
    g = (h & 0x00FF00) >> 8
    b =  h & 0x0000FF
    return (b, g, r)  # OpenCV uses BGR

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

detector = VehicleDetector()
tracker  = VehicleTracker()

source = cfg["video"]["source"]
cap = cv2.VideoCapture(source)

print("Running tracking — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks     = tracker.update(detections, frame)

    for track_id, x1, y1, x2, y2, cls in tracks:
        color = get_color(track_id)
        label = f"ID:{track_id} {CLASS_NAMES.get(cls, 'vehicle')}"

        # Thicker box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Bigger font
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Label background — sits above the box
        pad = 8
        cv2.rectangle(frame,
                      (x1, y1 - th - pad * 2),
                      (x1 + tw + pad * 2, y1),
                      color, -1)

        cv2.putText(frame, label,
                    (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness)

    cv2.putText(frame, f"Tracking {len(tracks)} vehicles", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    display = cv2.resize(frame, (1980, int(frame.shape[0] * 1980 / frame.shape[1])))
    cv2.imshow("Phase 3 - Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")