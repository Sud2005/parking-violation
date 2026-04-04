import yaml
import cv2
import numpy as np
from shapely.geometry import Polygon, Point

class ROIZone:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.zone_points = cfg["roi"]["zone"]                  # list of [x, y] pairs
        self.polygon     = Polygon(self.zone_points)           # Shapely polygon for checks
        self.np_points   = np.array(self.zone_points, np.int32)  # NumPy array for drawing

    def is_inside(self, x1, y1, x2, y2):
        """
        Check if a bounding box is inside the ROI.
        Uses the bottom-center of the box as the vehicle's ground contact point.
        This is more accurate than centroid for vehicles on a road.
        """
        bottom_center_x = (x1 + x2) // 2
        bottom_center_y = y2          # bottom edge of bounding box

        point = Point(bottom_center_x, bottom_center_y)
        return self.polygon.contains(point)

    def get_bottom_center(self, x1, y1, x2, y2):
        """Returns the ground contact point of a bounding box."""
        return ((x1 + x2) // 2, y2)

    def draw(self, frame, any_violation=False):
        """
        Draw the ROI zone on the frame.
        Turns red if any vehicle is violating.
        """
        overlay = frame.copy()

        # Filled polygon (semi-transparent)
        fill_color = (0, 0, 180) if any_violation else (0, 180, 0)
        cv2.fillPoly(overlay, [self.np_points], fill_color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Polygon border
        border_color = (0, 0, 255) if any_violation else (0, 255, 0)
        cv2.polylines(frame, [self.np_points], isClosed=True, color=border_color, thickness=2)

        # Zone label
        label     = "NO PARKING ZONE"
        label_pos = (self.np_points[:, 0].min(), self.np_points[:, 1].min() - 10)
        cv2.putText(frame, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

        return frame