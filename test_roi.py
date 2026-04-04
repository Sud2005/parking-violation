import cv2
import yaml
from detector import VehicleDetector
from tracker  import VehicleTracker
from roi import ROIZone

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def get_color(track_id):
    import hashlib
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    return ((h & 0xFF0000) >> 16), ((h & 0x00FF00) >> 8), (h & 0x0000FF)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

detector = VehicleDetector()
tracker  = VehicleTracker()
roi      = ROIZone()

source = cfg["video"]["source"]
cap    = cv2.VideoCapture(source)

print("Running ROI check — press Q to quit")

# In test_roi.py, add this before the while loop
DETECT_EVERY_N_FRAMES = 3   # run detection every 3rd frame, tune up/down
frame_count = 0
tracks = []                 # keep last known tracks between detection frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run detection + tracking on every Nth frame
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)

    # Drawing and ROI check runs every frame using last known tracks
    any_inside = any(roi.is_inside(x1, y1, x2, y2)
                     for _, x1, y1, x2, y2, _ in tracks)
    roi.draw(frame, any_violation=any_inside)
    # ... rest of your drawing code



    # Draw ROI zone (turns red if any vehicle is inside)
    roi.draw(frame, any_violation=any_inside)

    for track_id, x1, y1, x2, y2, cls in tracks:
        inside = roi.is_inside(x1, y1, x2, y2)

        # Box color — red if inside ROI, green if outside
        color = (0, 0, 255) if inside else (0, 255, 0)
        label = f"ID:{track_id} {CLASS_NAMES.get(cls, 'veh')[:3].upper()}"
        if inside:
            label += " [IN ZONE]"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw label background + text
        font_scale = 1.0
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 8
        cv2.rectangle(frame,
                       (x1, y1 - th - pad * 2),
                       (x1 + tw + pad * 2, y1),
                       color, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Draw ground contact point
        bx, by = roi.get_bottom_center(x1, y1, x2, y2)
        cv2.circle(frame, (bx, by), 6, color, -1)

    # Status bar at top
    status = f"Vehicles in zone: {sum(1 for _, x1,y1,x2,y2,_ in tracks if roi.is_inside(x1,y1,x2,y2))}"
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    display = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
    cv2.imshow("Phase 4 - ROI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")