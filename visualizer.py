import cv2

CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def get_color(track_id):
    import hashlib
    h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
    return ((h & 0xFF0000) >> 16), ((h & 0x00FF00) >> 8), (h & 0x0000FF)

def draw_vehicle(frame, track_id, x1, y1, x2, y2, cls,
                 inside, elapsed, violated):
    """Draw bounding box, label, and ground dot for one vehicle."""

    if violated:
        color = (0, 0, 255)       # red
    elif inside:
        color = (0, 165, 255)     # orange
    else:
        color = (0, 255, 0)       # green

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Label text
    cls_name = CLASS_NAMES.get(cls, "vehicle")[:3].upper()
    label    = f"ID:{track_id} {cls_name}"
    if inside:
        label += f"  {elapsed}s"
    if violated:
        label += "  VIOLATION"

    # Label background + text
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
    bx = (x1 + x2) // 2
    cv2.circle(frame, (bx, y2), 6, color, -1)

    # Flashing VIOLATION banner on the box itself
    if violated:
        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), (0, 0, 255), -1)
        cv2.putText(frame, "!! VIOLATION !!", (x1 + 6, y2 - 8),
                    font, 0.65, (255, 255, 255), 2)

def draw_status_bar(frame, in_zone, violation_count,
                    total_logged, threshold):
    """Draw the top status bar."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (20, 20, 20), -1)
    status = (f"In zone: {in_zone}   "
              f"Active violations: {violation_count}   "
              f"Total logged: {total_logged}   "
              f"Threshold: {threshold}s")
    cv2.putText(frame, status, (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)