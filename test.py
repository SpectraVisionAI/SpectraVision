from ObjectDetector import ObjectDetector
from VideoStream import VideoStream
from main import Application
import os


def run_test(video_path, model_path):
    detector = ObjectDetector(model_path=model_path)
    stream = VideoStream(video_path)
    app = Application(video_stream=stream, detector=detector)
    app.run()
    return detector.object_count


def test_models_on_videos():
    test_cases = [
        {"video": "vids/IMG_9747.MOV", "expected": 50},
        {"video": "vids/IMG_9746.MOV", "expected": 27}
    ]

    models = {
        "YOLO11n": "models/yolo11n.pt",
        "YOLO12n": "models/yolo12n.pt"
    }

    results = []

    for test in test_cases:
        for model_name, model_path in models.items():
            print(f"Testing {model_name} on {test['video']}...")

            if not os.path.exists(test["video"]):
                print(f"Video not found: {test['video']}")
                continue

            count = run_test(test["video"], test["expected"], model_path)
            error = abs(count - test["expected"])
            percent_error = (error / test["expected"]) * 100

            results.append({
                "video": test["video"],
                "model": model_name,
                "expected": test["expected"],
                "detected": count,
                "error": error,
                "percent_error": round(percent_error, 2)
            })

    print("\n--- Summary ---")
    for res in results:
        print(f"{res['model']} on {res['video']}: "
              f"Expected={res['expected']}, Detected={res['detected']}, "
              f"Error={res['error']}, Error%={res['percent_error']}%")


if __name__ == "__main__":
    test_models_on_videos()
