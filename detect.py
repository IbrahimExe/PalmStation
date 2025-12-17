import threading

# pip install opencv-python ultralytics
import cv2
from ultralytics.models import YOLO


class Detector:
    def __init__(self, model):
        self.model = model
        self.frame = None
        self.results = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._detect)
        self.thread.start()

    def _detect(self):
        while self.running:
            with self.lock:
                frame = self.frame
            if frame is not None:
                results = self.model(frame, verbose=False)
                with self.lock:
                    self.results = results

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def get_results(self):
        with self.lock:
            return self.results

    def stop(self):
        self.running = False
        self.thread.join()


def main():
    print("Loading model...")
    model = YOLO("yolov8x-oiv7.pt")

    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Opening window...")
    window_name = "Object Detection - Press 'q' to quit"

    detector = Detector(model)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Send frame to detector thread
        detector.update_frame(frame)

        # Get latest results
        results = detector.get_results()
        if results is None:
            continue

        # Draw results
        annotated_frame = results[0].plot(img=frame)
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == "__main__":
    main()