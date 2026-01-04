import time
from pathlib import Path
import cv2
from picamera2 import Picamera2
from common.resize_with_padding import resize_with_padding

camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(
        main={
            "format": "RGB888",  # This outputs BGR for OpenCV/YOLO
            "size": (1920, 1080),
        }
    )
)

camera.start()
print("Camera started.")

# Wait for camera to settle
time.sleep(2)

frame = camera.capture_array()

cv2.imwrite(str(Path(__file__).parent.parent / "fixture_image_full.jpg"), frame)
print("Frame saved as fixture_image_full.jpg")

resized_frame, scale = resize_with_padding(frame)

cv2.imwrite(
    str(Path(__file__).parent.parent / "fixture_image_resized.jpg"), resized_frame
)
print("Resized frame saved as fixture_image_resized.jpg")

camera.stop()
print("Camera stopped.")
