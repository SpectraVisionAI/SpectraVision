from ObjectDetector import ObjectDetector
from VideoStream import VideoStream
import cv2
import psutil
import os
from datetime import datetime


class Application:
    def __init__(self):
        self.detector = ObjectDetector()
        self.video_stream = VideoStream() # Change Source to 1 or 2 if 0 is not working
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()

    def get_memory_usage(self):
        mem = self.process.memory_info()
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        return f"Time: {elapsed:.1f}s, Memory: {mem.rss / 1024 / 1024:.1f} MB"

    def run(self):

        self.video_stream.setup()

        while True:
            ret, frame = self.video_stream.read()
            if not ret:
                break

            detection = self.detector.detect(frame)

            # Print memory usage every 5 seconds
            if int(datetime.now().timestamp()) % 5 == 0:
                print(self.get_memory_usage())

            cv2.imshow("Object Detection", detection)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    app.run()