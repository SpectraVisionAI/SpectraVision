from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO, solutions


class ObjectDetector:
    def __init__(self, model_path="models/yolo11n.pt", conf=0.6):
        """
        Initializes the ObjectDetector class by loading the YOLO11 model.

        Parameters:
          model_path (str): Path to the YOLO11 model file (e.g., "models/yolo11n.pt").
        """
        self.model = YOLO(model_path)
        self.object_count = 0
        self.tracked_objects = {}
        self.objects_left = defaultdict(int)

        self.regions_points = [(520, 0), (560, 0), (560, 720), (520, 720)]
        self.region_points = np.array(self.regions_points, dtype=np.int32)

    def point_in_region(self, point):
        return cv2.pointPolygonTest(self.region_points, point, False) >= 0

    def process_frame(self, frame):
        results = self.model.track(frame, persist=True, classes=[0,2], conf=0.6)[0]

        if results.boxes.id is not None:
            for box, track_id, cls in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])
                track_id = int(track_id)
                class_name = self.model.names[int(cls)]

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        "last_position": center,
                        "in_region": self.point_in_region(center),
                        "class": class_name
                    }
                else:
                    last_in_region = self.tracked_objects[track_id]["in_region"]
                    current_in_region = self.point_in_region(center)

                    if last_in_region and not current_in_region:
                        self.object_count += 1
                        self.objects_left[class_name] += 1

                    self.tracked_objects[track_id]["last_position"] = center
                    self.tracked_objects[track_id]["in_region"] = current_in_region

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.polylines(frame, [self.region_points], True, (255, 0, 0), 2)

        cv2.putText(frame, f"Objects Left: {self.object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        y_offset = 70
        for obj, count in self.objects_left.items():
            cv2.putText(frame, f"{obj}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30

        return frame
