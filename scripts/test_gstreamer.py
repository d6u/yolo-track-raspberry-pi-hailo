#!/usr/bin/env python3
"""Test GStreamer availability and OpenCV GStreamer backend support."""

import cv2
import subprocess
import shutil


def check_gstreamer_cli():
    """Check if GStreamer CLI tools are installed."""
    print("=" * 50)
    print("1. Checking GStreamer CLI tools...")
    print("=" * 50)

    # Check for gst-launch-1.0
    gst_launch = shutil.which("gst-launch-1.0")
    if gst_launch:
        print(f"✓ gst-launch-1.0 found at: {gst_launch}")
    else:
        print("✗ gst-launch-1.0 NOT found")

    # Check for gst-inspect-1.0
    gst_inspect = shutil.which("gst-inspect-1.0")
    if gst_inspect:
        print(f"✓ gst-inspect-1.0 found at: {gst_inspect}")
    else:
        print("✗ gst-inspect-1.0 NOT found")

    return gst_launch is not None


def check_opencv_gstreamer():
    """Check if OpenCV was built with GStreamer support."""
    print("\n" + "=" * 50)
    print("2. Checking OpenCV GStreamer support...")
    print("=" * 50)

    build_info = cv2.getBuildInformation()

    # Look for GStreamer in build info
    gstreamer_supported = "GStreamer" in build_info

    # Parse the GStreamer line
    for line in build_info.split("\n"):
        if "GStreamer" in line:
            print(f"OpenCV build info: {line.strip()}")
            if "YES" in line.upper():
                print("✓ OpenCV was built WITH GStreamer support")
                return True
            else:
                print("✗ OpenCV was built WITHOUT GStreamer support")
                return False

    print("✗ GStreamer not mentioned in OpenCV build info")
    return False


def check_gstreamer_plugins():
    """Check for required GStreamer plugins."""
    print("\n" + "=" * 50)
    print("3. Checking GStreamer plugins...")
    print("=" * 50)

    plugins_to_check = [
        ("x265enc", "H.265 encoder (libx265)"),
        ("x264enc", "H.264 encoder (libx264)"),
        ("mp4mux", "MP4 muxer"),
        ("videoconvert", "Video converter"),
        ("appsrc", "Application source"),
        ("filesink", "File sink"),
    ]

    available_encoders = []

    for plugin, description in plugins_to_check:
        try:
            result = subprocess.run(
                ["gst-inspect-1.0", plugin],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                print(f"✓ {plugin}: {description}")
                if "enc" in plugin:
                    available_encoders.append(plugin)
            else:
                print(f"✗ {plugin}: NOT FOUND - {description}")
        except FileNotFoundError:
            print(f"? {plugin}: Cannot check (gst-inspect-1.0 not available)")
        except subprocess.TimeoutExpired:
            print(f"? {plugin}: Check timed out")

    return available_encoders


def test_simple_pipeline():
    """Test a simple GStreamer pipeline with OpenCV."""
    print("\n" + "=" * 50)
    print("4. Testing simple GStreamer pipeline with OpenCV...")
    print("=" * 50)

    # Test with videotestsrc (synthetic video, no camera needed)
    test_pipeline = "videotestsrc num-buffers=10 ! videoconvert ! appsink"

    cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ GStreamer read pipeline works! Frame shape: {frame.shape}")
            cap.release()
            return True
        else:
            print("✗ Pipeline opened but could not read frame")
    else:
        print("✗ Could not open GStreamer read pipeline")

    cap.release()
    return False


def test_write_pipeline():
    """Test GStreamer write pipeline with OpenCV."""
    print("\n" + "=" * 50)
    print("5. Testing GStreamer write pipeline with OpenCV...")
    print("=" * 50)

    import numpy as np
    import os

    test_file = "/tmp/gstreamer_test.mp4"

    # Try different encoder pipelines
    pipelines = [
        (
            "x264enc",
            f"appsrc ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location={test_file}",
        ),
        (
            "x265enc",
            f"appsrc ! videoconvert ! x265enc speed-preset=ultrafast ! mp4mux ! filesink location={test_file}",
        ),
        (
            "openh264enc",
            f"appsrc ! videoconvert ! openh264enc ! mp4mux ! filesink location={test_file}",
        ),
    ]

    for encoder_name, pipeline in pipelines:
        print(f"\nTrying {encoder_name}...")

        # Clean up any existing test file
        if os.path.exists(test_file):
            os.remove(test_file)

        writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30, (640, 480))

        if writer.isOpened():
            print(f"  ✓ {encoder_name} pipeline opened successfully!")

            # Write a few test frames
            for i in range(30):
                # Create a test frame (blue gradient)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, 0] = i * 8  # Blue channel
                writer.write(frame)

            writer.release()

            # Check if file was created
            if os.path.exists(test_file):
                size = os.path.getsize(test_file)
                print(f"  ✓ Output file created: {test_file} ({size} bytes)")
                os.remove(test_file)
                return encoder_name
            else:
                print(f"  ✗ Output file was not created")
        else:
            print(f"  ✗ {encoder_name} pipeline failed to open")

    return None


def main():
    print("\n" + "=" * 50)
    print("GStreamer Availability Test")
    print("=" * 50)

    cli_ok = check_gstreamer_cli()
    opencv_ok = check_opencv_gstreamer()
    encoders = check_gstreamer_plugins() if cli_ok else []
    read_ok = test_simple_pipeline()
    working_encoder = test_write_pipeline()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if not cli_ok:
        print("\n✗ GStreamer CLI tools not installed.")
        print("  Install with: sudo apt install gstreamer1.0-tools")

    if not opencv_ok:
        print("\n✗ OpenCV not built with GStreamer support.")
        print("  You may need to rebuild OpenCV or install a different package.")
        print("  Try: pip install opencv-python-headless")
        print("  Or for full support, build OpenCV from source with GStreamer enabled.")

    if cli_ok and not encoders:
        print("\n✗ No video encoders found.")
        print(
            "  Install with: sudo apt install gstreamer1.0-plugins-ugly gstreamer1.0-plugins-bad"
        )
        print("  For x265: sudo apt install gstreamer1.0-plugins-bad")
        print("  For x264: sudo apt install gstreamer1.0-plugins-ugly")

    if working_encoder:
        print(f"\n✓ Working encoder found: {working_encoder}")
        print(f"  Use this pipeline:")
        if working_encoder == "x264enc":
            print(
                '  gst_out = "appsrc ! videoconvert ! x264enc tune=zerolatency ! mp4mux ! filesink location=output.mp4"'
            )
        elif working_encoder == "x265enc":
            print(
                '  gst_out = "appsrc ! videoconvert ! x265enc speed-preset=ultrafast ! mp4mux ! filesink location=output.mp4"'
            )
    elif opencv_ok and read_ok:
        print("\n✗ No working write pipeline found.")
        print("  GStreamer read works but write does not.")
        print("  Check encoder plugin installation.")
    else:
        print("\n✗ GStreamer integration with OpenCV is not working.")


if __name__ == "__main__":
    main()
