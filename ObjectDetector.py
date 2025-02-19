from ultralytics import YOLO, solutions

class ObjectDetector:
    def __init__(self, model_path="models/yolo11n.pt", conf=0.6):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.regions_points = [(520, 0), (560, 0), (560, 720), (520, 720)]
        self.counter = solutions.ObjectCounter(
            model=model_path,
            region=self.regions_points,
            line_width=2,
            classes=[0, 2],
            conf=conf,
            show_in=True,
            show_out=True,
            persist=True,
            tracker="tracker/bytetrack.yaml",
            stream=True
        )

    def process_image(self, frame):
        return self.counter.count(frame)


