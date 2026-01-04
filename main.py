import argparse
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from common import resize_with_padding
from common.tracker import SimpleTracker
from hailo_platform import (
    HEF,
    ConfigureParams,
    Device,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)


# INPUT_W = 1280
# INPUT_H = 720
INPUT_W = 1920
INPUT_H = 1080

# Class IDs for objects we want to log (from COCO labels)
LOGGED_CLASSES = {
    0: "person",
    15: "cat",
    16: "dog",
}


def log_detection(
    log_file: Path,
    video_filename: str,
    class_name: str,
    track_id: int | None,
    score: float,
):
    """
    Log a detection event to a text file.

    Args:
        log_file: Path to the log file
        video_filename: Current video file being recorded
        class_name: Name of the detected class (person/cat/dog)
        track_id: Track ID if tracking is enabled
        score: Detection confidence score
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_info = f" (Track ID: {track_id})" if track_id is not None else ""
    log_entry = f"{timestamp} | Video: {video_filename} | Detected: {class_name}{track_info} | Confidence: {score * 100:.1f}%\n"

    with open(log_file, "a") as f:
        f.write(log_entry)


def extract_detections(
    detections: list, img_height: int, img_width: int, score_threshold: float = 0.25
):
    """
    Extract detections from the model output.

    Args:
        detections: Raw detections from the model (list per class)
        img_height: Height of the original image
        img_width: Width of the original image
        score_threshold: Minimum confidence threshold

    Returns:
        dict: Detection results with boxes, classes, scores
    """
    size = max(img_height, img_width)
    padding_length = int(abs(img_height - img_width) / 2)

    all_detections = []

    for class_id, class_detections in enumerate(detections):
        for det in class_detections:
            bbox, score = det[:4], det[4]
            if score >= score_threshold:
                # Scale box coordinates
                box = [int(x * size) for x in bbox]

                # Apply padding correction and swap to [xmin, ymin, xmax, ymax]
                for i in range(4):
                    if i % 2 == 0:  # x-coordinates
                        if img_height != size:
                            box[i] -= padding_length
                    else:  # y-coordinates
                        if img_width != size:
                            box[i] -= padding_length

                # Convert to [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box[1], box[0], box[3], box[2]
                all_detections.append((score, class_id, [xmin, ymin, xmax, ymax]))

    # Sort by score descending and take top detections
    all_detections.sort(reverse=True, key=lambda x: x[0])
    top_detections = all_detections[:50]

    if top_detections:
        scores, class_ids, boxes = zip(*top_detections)
    else:
        scores, class_ids, boxes = [], [], []

    return {
        "boxes": list(boxes),
        "classes": list(class_ids),
        "scores": list(scores),
    }


def draw_detections(frame: np.ndarray, detections: dict, labels: list) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame: Image to draw on
        detections: Dict with boxes, classes, scores, and optionally track_ids
        labels: List of class labels

    Returns:
        Frame with detections drawn
    """
    # Generate colors for track IDs (more distinct colors for tracking)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)

    track_ids = detections.get("track_ids", [None] * len(detections["boxes"]))

    for box, class_id, score, track_id in zip(
        detections["boxes"], detections["classes"], detections["scores"], track_ids
    ):
        xmin, ymin, xmax, ymax = map(int, box)

        # Use track ID for color if available, otherwise use class ID
        color_idx = track_id % 200 if track_id is not None else class_id % 200
        color = tuple(map(int, colors[color_idx]))

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Prepare label text with track ID if available
        if track_id is not None:
            label = f"ID:{track_id} {labels[class_id]}: {score * 100:.1f}%"
        else:
            label = f"{labels[class_id]}: {score * 100:.1f}%"

        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame,
            (xmin, ymin - text_height - 6),
            (xmin + text_width + 4, ymin),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (xmin + 2, ymin - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def create_video_writer(
    output_dir: Path, width: int, height: int, fps: float = 30.0
) -> tuple[cv2.VideoWriter, Path]:
    """
    Create a new video writer with a timestamped filename.

    Args:
        output_dir: Directory to save video files
        width: Frame width
        height: Frame height
        fps: Frames per second

    Returns:
        Tuple of (VideoWriter, output_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"detection_{timestamp}.mp4"
    gst_out = (
        f"appsrc ! videoconvert ! "
        f'x265enc option-string="crf=28" speed-preset=fast tune=zerolatency ! '
        f"h265parse ! mp4mux ! filesink location={output_path}"
    )
    # gst_out = (
    #     f"appsrc ! videoconvert ! "
    #     f"x264enc pass=qual quantizer=28 speed-preset=fast tune=zerolatency ! "
    #     f"mp4mux ! filesink location={output_path}"
    # )
    writer = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if writer.isOpened():
        print(f"Recording to: {output_path}")
        return writer, output_path
    writer.release()
    raise RuntimeError("Cannot open VideoWriter")


def run_live_detection(
    hef_path: Path,
    display_temp=False,
    preview=True,
    enable_tracking=False,
    duration=None,
):
    with open(Path(__file__).parent / "common" / "coco_labels.txt", "r") as f:
        COCO_CLASSES = [line.strip() for line in f.readlines()]

    # Initialize tracker if enabled
    tracker = (
        SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        if enable_tracking
        else None
    )
    if enable_tracking:
        print("Object tracking enabled.")

    # Initialize camera
    camera = Picamera2()
    camera.configure(
        camera.create_preview_configuration(
            main={
                "format": "RGB888",  # This outputs BGR for OpenCV/YOLO
                "size": (INPUT_W, INPUT_H),
            }
        )
    )
    camera.start()
    print("Camera started.")

    # Wait for camera to settle
    time.sleep(2)

    # For getting device temperature
    device = Device() if display_temp else None

    # Setup inference
    vdevice = VDevice()
    hef = HEF(str(hef_path))

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=HailoStreamInterface.PCIe
    )
    network_group = vdevice.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()
    input_vstreams_params = InputVStreamParams.make(
        network_group, format_type=FormatType.UINT8
    )
    output_vstreams_params = OutputVStreamParams.make(
        network_group, format_type=FormatType.FLOAT32
    )

    # Video recording setup (when not in preview mode)
    video_writer = None
    video_start_time = None
    video_rotation_interval = 120  # seconds
    output_dir = Path(__file__).parent / "recordings"
    current_video_path: Path | None = None

    # Detection logging setup
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists for logging
    log_file = output_dir / "detections_log.txt"

    # Duration-based termination setup (based on video recording duration)
    if duration is not None:
        print(f"Script will terminate after {duration} seconds of video recording.")

    # (track_id, class_id) pairs already logged
    logged_tracks: set[tuple[int, int]] = set()
    # class_id -> last log time (for non-tracking mode)
    last_logged_time: dict[int, float] = {}
    log_cooldown = 5.0  # seconds between logging same class without tracking

    with InferVStreams(
        network_group, input_vstreams_params, output_vstreams_params
    ) as infer_pipeline:
        with network_group.activate(network_group_params):
            # FPS calculation variables
            frame_count = 0
            fps = 0.0
            fps_start_time = time.time()

            while True:
                original_frame = camera.capture_array()
                resized_frame, scale = resize_with_padding(original_frame)

                # Add batch dimension (1, H, W, C) for Hailo inference
                input_batch = np.expand_dims(resized_frame, axis=0)

                # Run inference
                infer_results = infer_pipeline.infer(input_batch)

                # Post-processing - Extract and draw detections
                for output_name, raw_detections in infer_results.items():
                    # Remove batch dimension and process detections
                    detections_data = raw_detections[0]

                    detections = extract_detections(
                        detections_data,
                        original_frame.shape[0],
                        original_frame.shape[1],
                        score_threshold=0.25,
                    )

                    # Apply tracking if enabled
                    if tracker is not None:
                        detections = tracker.update(detections)

                    # Log detections for cat, dog, person
                    video_filename = (
                        current_video_path.name
                        if current_video_path
                        else "preview_mode"
                    )
                    track_ids = detections.get(
                        "track_ids", [None] * len(detections["boxes"])
                    )
                    current_time_log = time.time()

                    for class_id, score, track_id in zip(
                        detections["classes"], detections["scores"], track_ids
                    ):
                        if class_id in LOGGED_CLASSES:
                            class_name = LOGGED_CLASSES[class_id]

                            if tracker is not None and track_id is not None:
                                # With tracking: log each unique track only once
                                track_key = (track_id, class_id)
                                if track_key not in logged_tracks:
                                    log_detection(
                                        log_file,
                                        video_filename,
                                        class_name,
                                        track_id,
                                        score,
                                    )
                                    logged_tracks.add(track_key)
                            else:
                                # Without tracking: use cooldown to avoid spam
                                last_time = last_logged_time.get(class_id, 0)
                                if current_time_log - last_time >= log_cooldown:
                                    log_detection(
                                        log_file,
                                        video_filename,
                                        class_name,
                                        None,
                                        score,
                                    )
                                    last_logged_time[class_id] = current_time_log

                    original_frame = draw_detections(
                        original_frame, detections, COCO_CLASSES
                    )

                # Display temperature information on frame if enabled
                if display_temp and device:
                    # Get live temperature information if enabled
                    ts0_temperature = (
                        device.control.get_chip_temperature().ts0_temperature
                    )
                    ts1_temperature = (
                        device.control.get_chip_temperature().ts1_temperature
                    )

                    temperature_text = f"Temperature: ts0: {ts0_temperature:.1f}C, ts1: {ts1_temperature:.1f}C"
                    cv2.putText(
                        original_frame,
                        temperature_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # Calculate FPS by counting frames per second
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(
                    original_frame,
                    fps_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                if preview:
                    # Display the frame
                    cv2.imshow("Live Detection", original_frame)

                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Save frames to video file
                    current_time = time.time()

                    # Create new video writer if needed (first frame or rotation)
                    if (
                        video_writer is None
                        or video_start_time is None
                        or (current_time - video_start_time) >= video_rotation_interval
                    ):
                        if video_writer is not None:
                            video_writer.release()
                            print("Video file rotated.")
                            # Clear logged tracks when video rotates (to re-log in new video)
                            logged_tracks.clear()
                        video_writer, current_video_path = create_video_writer(
                            output_dir,
                            original_frame.shape[1],
                            original_frame.shape[0],
                            fps=30.0,
                        )
                        video_start_time = current_time

                    # Write frame to video
                    video_writer.write(original_frame)

                    # Check for duration-based termination (based on video recording time)
                    if duration is not None and video_start_time is not None:
                        video_duration = current_time - video_start_time
                        if video_duration >= duration:
                            print(
                                f"Video duration of {duration} seconds reached. Terminating gracefully..."
                            )
                            break

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print("Video recording stopped.")
    camera.stop()
    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live detection with optional temperature monitoring"
    )
    parser.add_argument(
        "--display-temp",
        action="store_true",
        help="Display temperature information on frames",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a desktop window for live detection",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable object tracking across frames",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Run for specified seconds then terminate gracefully",
    )
    args = parser.parse_args()

    run_live_detection(
        Path(__file__).parent / "models" / "yolov11l.hef",
        display_temp=args.display_temp,
        preview=args.preview,
        enable_tracking=args.track,
        duration=args.duration,
    )

    print("Live detection stopped.")
