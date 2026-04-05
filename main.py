import cv2
import yaml
from detector  import VehicleDetector
from tracker   import VehicleTracker
from roi       import ROIZone
from temporal  import TemporalMonitor
from visualizer import draw_vehicle, draw_status_bar
from logger    import ViolationLogger

# ── Load config ──────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# ── Initialise all modules ────────────────────────────────────────────────────
detector = VehicleDetector()
tracker  = VehicleTracker()
roi      = ROIZone()
monitor  = TemporalMonitor()
logger   = ViolationLogger(cfg["output"]["log_path"])

# ── Video input ───────────────────────────────────────────────────────────────
source = cfg["video"]["source"]
cap    = cv2.VideoCapture(source)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30
W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── Video output (optional) ───────────────────────────────────────────────────
writer = None
if cfg["output"]["save_video"]:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cfg["output"]["output_path"], fourcc, fps, (W, H))
    print(f"Saving output to: {cfg['output']['output_path']}")

# ── Main loop ─────────────────────────────────────────────────────────────────
DETECT_EVERY = 3
frame_count  = 0
tracks       = []

print("Running — press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detection + tracking (every Nth frame for speed)
    if frame_count % DETECT_EVERY == 0:
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame)

    # Purge tracks that vanished without leaving the ROI
    active_ids = {t[0] for t in tracks}
    monitor.purge_stale(active_ids)

    violation_count = 0

    for track_id, x1, y1, x2, y2, cls in tracks:
        inside            = roi.is_inside(x1, y1, x2, y2)
        elapsed, violated = monitor.update(track_id, inside)

        # Log new violations to CSV (only logs once per track_id)
        if violated:
            from detector import VehicleDetector
            cls_names = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
            logger.log(
                track_id  = track_id,
                vehicle_class = cls_names.get(cls, "vehicle"),
                elapsed   = elapsed,
                bbox      = (x1, y1, x2, y2)
            )
            violation_count += 1

        # Reset logger entry when vehicle leaves so it can be re-logged
        if not inside:
            logger.reset_id(track_id)

        draw_vehicle(frame, track_id, x1, y1, x2, y2,
                     cls, inside, elapsed, violated)

    # Draw ROI zone
    roi.draw(frame, any_violation=violation_count > 0)

    # Draw status bar
    draw_status_bar(
        frame,
        in_zone         = len(monitor.get_all_elapsed()),
        violation_count = violation_count,
        total_logged    = logger.total_logged(),
        threshold       = cfg["violation"]["time_threshold_seconds"]
    )

    # Save frame to output video
    if writer:
        writer.write(frame)

    # Display
    display = cv2.resize(frame, (1280, int(H * 1280 / W)))
    cv2.imshow("Parking Violation Detection", display)

    delay = max(1, int(1000 / fps))
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()

print(f"\nDone. Total violations logged: {logger.total_logged()}")
print(f"CSV saved to: {cfg['output']['log_path']}")
if cfg['output']['save_video']:
    print(f"Video saved to: {cfg['output']['output_path']}")