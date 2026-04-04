import yaml
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, config_path="config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        trk = cfg["tracking"]
        self.tracker = DeepSort(
            max_age=trk["max_age"],       # frames to keep a lost track alive
            n_init=3,                      # frames needed to confirm a new track
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,       # appearance similarity threshold
            nn_budget=None,
        )

    def update(self, detections, frame):
        """
        Takes raw detections from VehicleDetector and returns confirmed tracks.

        Input detections: [[x1,y1,x2,y2, conf, cls], ...]
        Returns tracks:   [(track_id, x1, y1, x2, y2, cls), ...]
        """
        # DeepSORT expects: [([x1,y1,w,h], confidence, class), ...]
        ds_input = []
        for x1, y1, x2, y2, conf, cls in detections:
            w = x2 - x1
            h = y2 - y1
            ds_input.append(([x1, y1, w, h], conf, cls))

        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame)

        tracks = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue                   # skip tentative tracks

            track_id = track.track_id
            l, t, w, h = track.to_ltwh()
            x1 = int(l)
            y1 = int(t)
            x2 = int(l + w)
            y2 = int(t + h)
            cls = int(track.det_class) if track.det_class is not None else 2

            tracks.append((track_id, x1, y1, x2, y2, cls))

        return tracks