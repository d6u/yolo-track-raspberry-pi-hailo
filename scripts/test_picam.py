from picamera2 import Picamera2

camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
)
camera.start()
print("Camera started.")

frame = camera.capture_array()
print("Frame captured, shape:", frame.shape)

camera.stop()
print("Camera stopped.")
