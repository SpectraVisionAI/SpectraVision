from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, ):
        self.model = YOLO("models/yolo11n.pt")

    def detect(self, frame):
        results = self.model.track(frame)
        annotated_frame = results[0].plot()

        return annotated_frame


