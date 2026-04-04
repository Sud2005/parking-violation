import cv2
import yaml
import hashlib
from detector  import VehicleDetector
from tracker   import VehicleTracker
from roi       import ROIZone
from temporal  import TemporalMonitor

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def get_color(track_id):
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    return ((h & 0xFF0000) >> 16), ((h & 0x00FF00) >> 8), (h & 0x0000FF)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

detector = VehicleDetector()
tracker  = VehicleTracker()
roi      = ROIZone()
monitor  = TemporalMonitor()

DETECT_EVERY_N_FRAMES = 3
frame_count = 0
tracks      = []

source = cfg["video"]["source"]
cap    = cv2.VideoCapture(source)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30

print(f"Threshold: {cfg['violation']['time_threshold_seconds']}s — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)

    # IDs currently visible
    active_ids = {t[0] for t in tracks}
    monitor.purge_stale(active_ids)

    violations = []

    for track_id, x1, y1, x2, y2, cls in tracks:
        inside             = roi.is_inside(x1, y1, x2, y2)
        elapsed, violated  = monitor.update(track_id, inside)

        # Box color logic
        if violated:
            color = (0, 0, 255)      # red   — violation
        elif inside:
            color = (0, 165, 255)    # orange — in zone, timer running
        else:
            color = (0, 255, 0)      # green  — outside zone

        # Build label
        label = f"ID:{track_id} {CLASS_NAMES.get(cls,'veh')[:3].upper()}"
        if inside:
            label += f"  {elapsed}s"
        if violated:
            label += "  VIOLATION"
            violations.append(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Draw label background + text
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thick = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        pad = 6
        cv2.rectangle(frame,
                       (x1, y1 - th - pad * 2),
                       (x1 + tw + pad * 2, y1),
                       color, -1)
        cv2.putText(frame, label, (x1 + pad, y1 - pad),
                    font, scale, (255, 255, 255), thick)

        # Ground contact dot
        bx, by = roi.get_bottom_center(x1, y1, x2, y2)
        cv2.circle(frame, (bx, by), 6, color, -1)

    # Draw ROI zone
    any_violation = len(violations) > 0
    roi.draw(frame, any_violation=any_violation)

    # Status panel at top
    elapsed_map = monitor.get_all_elapsed()
    in_zone     = len(elapsed_map)
    status      = f"In zone: {in_zone}  |  Violations: {len(violations)}  |  Threshold: {cfg['violation']['time_threshold_seconds']}s"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(frame, status, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    display = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
    cv2.imshow("Phase 5 - Temporal Monitor", display)

    delay = max(1, int(1000 / fps))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")