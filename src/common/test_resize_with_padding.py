import unittest
import cv2
import numpy as np
from .resize_with_padding import *
from pathlib import Path


class TestResizeWithPadding(unittest.TestCase):
    def setUp(self):
        self.fixture_full = cv2.imread(
            str(Path(__file__).parent / "fixture_image_full.jpg")
        ).astype(np.uint8)

        self.fixture_resized = cv2.imread(
            str(Path(__file__).parent / "fixture_image_resized.jpg")
        ).astype(np.uint8)

    def test_resize_with_padding(self):
        resized_frame, scale = resize_with_padding(
            self.fixture_full, target_size=(640, 640)
        )

        self.assertEqual(
            resized_frame.shape,
            (640, 640, 3),
            f"Expected (640, 640, 3), got {resized_frame.shape}",
        )

        # Allow small differences due to JPEG compression and interpolation
        max_diff = np.abs(
            resized_frame.astype(np.int16) - self.fixture_resized.astype(np.int16)
        ).max()

        self.assertTrue(
            np.allclose(resized_frame, self.fixture_resized, atol=30),
            f"resized_frame and fixture_resized differ too much (max diff: {max_diff})",
        )


if __name__ == "__main__":
    unittest.main()
