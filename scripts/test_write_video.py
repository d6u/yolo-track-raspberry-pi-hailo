import time
import cv2
from picamera2 import Picamera2

# Settings for 1080p
width = 1920
height = 1080
fps = 30

# GStreamer Pipeline for MINIMUM SPACE
# - x265enc: H.265 encoder (needs h265parse for mp4mux)
# - x264enc: H.264 encoder (works directly with mp4mux)
# - bitrate: 2000 kbps is usually plenty for 1080p30 (adjust lower for smaller files)

# Option 1: H.265 (better compression, needs h265parse)
gst_out = (
    f"appsrc ! videoconvert ! "
    f"x265enc speed-preset=ultrafast bitrate=2000 ! h265parse ! "
    f"mp4mux ! filesink location=output_h265.mp4"
)

# Option 2: H.264 (more compatible)
# gst_out = (
#     f"appsrc ! videoconvert ! "
#     f"x264enc tune=zerolatency bitrate=2000 ! "
#     f"mp4mux ! filesink location=output_h264.mp4"
# )

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

# Recording settings
duration_seconds = 10
total_frames = fps * duration_seconds

# Initialize VideoWriter with GStreamer backend
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height))

if out.isOpened():
    print(f"Recording {duration_seconds} seconds of video...")

    for frame_count in range(total_frames):
        frame = camera.capture_array()
        out.write(frame)

        # Print progress every second
        if (frame_count + 1) % fps == 0:
            elapsed = (frame_count + 1) // fps
            print(f"  {elapsed}/{duration_seconds} seconds recorded")

    print("Recording complete!")
else:
    print("Error: Could not open video writer")

out.release()
camera.stop()
print("Done. Video saved to output_h264.mp4")
