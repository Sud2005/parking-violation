import time
import yaml

class TemporalMonitor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.threshold = cfg["violation"]["time_threshold_seconds"]

        # {track_id: {"start_time": float, "last_seen": float}}
        self.active_tracks = {}

    def update(self, track_id, inside_roi):
        """
        Call this every frame for every tracked vehicle.
        Returns (elapsed_seconds, is_violation)
        """
        now = time.time()

        if inside_roi:
            if track_id not in self.active_tracks:
                # Vehicle just entered the zone — start the clock
                self.active_tracks[track_id] = {
                    "start_time": now,
                    "last_seen":  now
                }
            else:
                # Vehicle still in zone — update last seen
                self.active_tracks[track_id]["last_seen"] = now

            elapsed      = now - self.active_tracks[track_id]["start_time"]
            is_violation = elapsed >= self.threshold
            return round(elapsed, 1), is_violation

        else:
            # Vehicle left the zone — clean up
            if track_id in self.active_tracks:
                del self.active_tracks[track_id]
            return 0.0, False

    def get_all_elapsed(self):
        """Returns {track_id: elapsed_seconds} for all vehicles currently in zone."""
        now = time.time()
        return {
            tid: round(now - data["start_time"], 1)
            for tid, data in self.active_tracks.items()
        }

    def purge_stale(self, active_track_ids, timeout=2.0):
        """
        Remove tracks that haven't been seen for `timeout` seconds.
        Call this every frame to clean up vehicles that left without
        triggering the outside-ROI path (e.g. video skip or occlusion).
        """
        now  = time.time()
        stale = [
            tid for tid, data in self.active_tracks.items()
            if tid not in active_track_ids
            and (now - data["last_seen"]) > timeout
        ]
        for tid in stale:
            del self.active_tracks[tid]