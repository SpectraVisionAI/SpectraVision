import time

from ObjectDetector import ObjectDetector
from VideoStream import VideoStream
import cv2
import psutil
import os
from datetime import datetime

class Application:
    def __init__(self):
        self.detector = ObjectDetector()
        self.video_stream = VideoStream("vids/IMG_9746.MOV")  # Change source if needed
        self.process = psutil.Process(os.getpid())
        self.start_time = datetime.now()
        self.last_memory_print = 0

    def get_memory_usage(self):
        mem = self.process.memory_info()
        current_time = datetime.now()
        elapsed = (current_time - self.start_time).total_seconds()
        return f"Time: {elapsed:.1f}s, Memory: {mem.rss / 1024 / 1024:.1f} MB"

    def run(self):
        self.video_stream.setup()
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = self.video_stream.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

            frame = cv2.resize(frame, (1080, 720))
            detection = self.detector.process_frame(frame)

            cv2.imshow("Detection", detection)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = Application()
    app.run()