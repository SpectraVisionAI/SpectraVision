import unittest
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
from main import Application
from VideoStream import VideoStream
from ObjectDetector import ObjectDetector

class TestApplication(unittest.TestCase):

    @patch("VideoStream.VideoStream")
    @patch("ObjectDetector.ObjectDetector")
    def setUp(self, MockObjectDetector, MockVideoStream):
        self.mock_detector = MockObjectDetector.return_value
        self.mock_video_stream = MockVideoStream.return_value

        self.mock_video_stream.read.side_effect = [(True, np.zeros((720, 1080, 3), dtype=np.uint8))] * 10 + [(False, None)]
        self.mock_video_stream.setup.return_value = None
        self.mock_video_stream.release.return_value = None

        self.mock_detector.process_frame.side_effect = lambda frame: frame

        self.app = Application(video_stream=self.mock_video_stream, detector=self.mock_detector)


    def test_video_stream_setup(self):
        self.app.video_stream.setup()
        self.mock_video_stream.setup.assert_called_once()

    def test_video_stream_read(self):
        self.app.video_stream.setup()
        ret, frame = self.app.video_stream.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)

    def test_video_stream_release(self):
        self.app.video_stream.setup()
        self.app.video_stream.release()
        self.mock_video_stream.release.assert_called_once()



    @patch("cv2.imshow")
    @patch("cv2.waitKey", return_value=ord('q'))
    def test_application_run(self, mock_waitKey, mock_imshow):
        self.app.run()
        self.mock_video_stream.read.assert_called()
        self.mock_video_stream.release.assert_called_once()

if __name__ == "__main__":
    unittest.main()
