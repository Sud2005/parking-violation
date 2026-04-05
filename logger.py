import csv
import os
from datetime import datetime

class ViolationLogger:
    def __init__(self, log_path="data/violations.csv"):
        self.log_path    = log_path
        self.logged_ids  = set()   # track which IDs have already been logged

        # Create CSV with headers if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "violation_id",
                    "track_id",
                    "vehicle_class",
                    "elapsed_seconds",
                    "timestamp",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
                ])

    def log(self, track_id, vehicle_class, elapsed, bbox):
        """
        Log a violation only once per track_id.
        Returns True if this is a new log entry, False if already logged.
        """
        if track_id in self.logged_ids:
            return False

        self.logged_ids.add(track_id)
        x1, y1, x2, y2 = bbox

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                len(self.logged_ids),          # violation_id (auto increment)
                track_id,
                vehicle_class,
                elapsed,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                x1, y1, x2, y2
            ])

        print(f"[VIOLATION LOGGED] ID:{track_id} | {vehicle_class} | {elapsed}s")
        return True

    def reset_id(self, track_id):
        """
        Call this when a vehicle leaves the zone so it can be re-logged
        if it re-enters and violates again.
        """
        self.logged_ids.discard(track_id)

    def total_logged(self):
        return len(self.logged_ids)