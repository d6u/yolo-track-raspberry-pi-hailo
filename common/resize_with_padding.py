import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


def resize_with_padding(
    frame: NDArray[np.uint8], target_size: Tuple[int, int] = (640, 640)
) -> Tuple[NDArray[np.uint8], float]:
    """
    Resize the frame to fit within target_size while maintaining aspect ratio,
    and pad with black to reach target_size.

    Returns:
        A tuple of (resized_padded_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas, scale
