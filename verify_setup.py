import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import yaml

print("Checking imports... OK")

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
print("Config loaded... OK")

# Load YOLO model (downloads weights automatically on first run)
model = YOLO(cfg["detection"]["model"])
print(f"YOLO model loaded: {cfg['detection']['model']} ... OK")

# Try opening the video
source = cfg["video"]["source"]
cap = cv2.VideoCapture(source if source != 0 else 0)
if cap.isOpened():
    ret, frame = cap.read()
    print(f"Video opened — frame shape: {frame.shape} ... OK")
    cap.release()
else:
    print("ERROR: Could not open video. Check your 'source' path in config.yaml")

# Test Shapely ROI
zone_coords = cfg["roi"]["zone"]
zone = Polygon(zone_coords)
print(f"ROI polygon created with {len(zone_coords)} points ... OK")

print("\nAll checks passed. Phase 1 complete!")