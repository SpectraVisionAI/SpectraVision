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

    def get_infos(self):
        w, h, fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        return w, h, fps

