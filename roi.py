# run_roi_picker.py
import cv2
import yaml

points = []
DISPLAY_WIDTH = 900  # adjust this to fit your screen

def click(event, x, y, flags, param):
    scale = param["scale"]
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coords back to original coords
        real_x = int(x / scale)
        real_y = int(y / scale)
        points.append([real_x, real_y])
        print(f"Point added: ({real_x}, {real_y})")
        cv2.circle(display_frame, (x, y), 6, (0, 255, 0), -1)
        if len(points) > 1:
            dx, dy = int(points[-2][0] * scale), int(points[-2][1] * scale)
            cv2.line(display_frame, (dx, dy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Pick ROI - press Q when done", display_frame)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

source = cfg["video"]["source"]
cap = cv2.VideoCapture(source)

# Skip to middle of video for a better reference frame
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read frame from video.")
    exit()

# Scale down to fit screen
h, w = frame.shape[:2]
scale = DISPLAY_WIDTH / w
display_h = int(h * scale)
display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_h))

print(f"Original resolution: {w}x{h}")
print(f"Display resolution:  {DISPLAY_WIDTH}x{display_h}  (scale: {scale:.2f})")
print("Click the corners of your no-parking zone in order, then press Q")

cv2.imshow("Pick ROI - press Q when done", display_frame)
cv2.setMouseCallback("Pick ROI - press Q when done", click, {"scale": scale})

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if len(points) < 3:
    print("Need at least 3 points to define a zone.")
else:
    print(f"\nYour zone coordinates (already scaled to original resolution):")
    print(points)
    print("\nPaste these into config.yaml under roi.zone")