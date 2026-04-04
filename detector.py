import cv2
import yaml
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        det = cfg["detection"]
        self.model = YOLO(det["model"])
        self.confidence = det["confidence"]
        self.vehicle_classes = det["vehicle_classes"]
        self.imgsz = det["imgsz"]

        self.min_box_area = {
            2: 2000,  # car — minimum pixel area
            3: 500,  # motorcycle
            5: 8000,  # bus
            7: 5000,  # truck
        }

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.vehicle_classes,
            imgsz=self.imgsz,
            verbose=False,

        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Reject if box is too small for its claimed class
            area = (x2 - x1) * (y2 - y1)
            min_area = self.min_box_area.get(cls, 0)
            if area < min_area:
                continue

            detections.append([x1, y1, x2, y2, conf, cls])

        return detections