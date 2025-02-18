import cv2


class VideoStream:
    def __init__(self):
        self.cap = None

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def setup(self):
        for index in range(0, 4):
            self.cap = cv2.VideoCapture(index)
            if self.cap.isOpened():
                print(f"Camera found at index {index}")
                break
        else:
            raise RuntimeError("No cameras detected")

    def release(self):
        self.cap.release()
