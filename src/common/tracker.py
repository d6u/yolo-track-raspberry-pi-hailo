"""
Simple Online and Realtime Tracking (SORT) implementation for object tracking.
"""

import numpy as np
from collections import defaultdict


def iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]

    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def linear_assignment(cost_matrix: np.ndarray) -> tuple:
    """
    Simple greedy assignment based on cost matrix.

    Args:
        cost_matrix: NxM cost matrix (negative IoU for minimization)

    Returns:
        Tuple of (matched_indices, unmatched_rows, unmatched_cols)
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(cost_matrix.shape[0])),
            list(range(cost_matrix.shape[1])),
        )

    matched_indices = []
    rows_available = set(range(cost_matrix.shape[0]))
    cols_available = set(range(cost_matrix.shape[1]))

    # Greedy assignment - always pick the best match
    while rows_available and cols_available:
        min_val = float("inf")
        best_match = None

        for i in rows_available:
            for j in cols_available:
                if cost_matrix[i, j] < min_val:
                    min_val = cost_matrix[i, j]
                    best_match = (i, j)

        if best_match is None or min_val >= 0:  # No good match found (IoU threshold)
            break

        matched_indices.append(best_match)
        rows_available.remove(best_match[0])
        cols_available.remove(best_match[1])

    unmatched_rows = list(rows_available)
    unmatched_cols = list(cols_available)

    return (
        np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int),
        unmatched_rows,
        unmatched_cols,
    )


class TrackedObject:
    """Represents a tracked object with state history."""

    _id_counter = 0

    def __init__(self, box: list, class_id: int, score: float):
        TrackedObject._id_counter += 1
        self.id = TrackedObject._id_counter
        self.box = box
        self.class_id = class_id
        self.score = score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = [box]

    def update(self, box: list, score: float):
        """Update track with new detection."""
        self.box = box
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.history.append(box)
        if len(self.history) > 30:
            self.history.pop(0)

    def predict(self):
        """Predict next position (simple - use last position)."""
        self.age += 1
        self.time_since_update += 1
        return self.box

    @classmethod
    def reset_id_counter(cls):
        """Reset the ID counter."""
        cls._id_counter = 0


class SimpleTracker:
    """
    Simple SORT-style tracker using IoU matching.
    """

    def __init__(
        self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3
    ):
        """
        Initialize tracker.

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[TrackedObject] = []
        self.frame_count = 0

    def update(self, detections: dict) -> dict:
        """
        Update tracks with new detections.

        Args:
            detections: Dict with boxes, classes, scores

        Returns:
            Dict with tracked boxes, classes, scores, and track_ids
        """
        self.frame_count += 1

        boxes = detections.get("boxes", [])
        classes = detections.get("classes", [])
        scores = detections.get("scores", [])

        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        if len(self.tracks) > 0 and len(boxes) > 0:
            # Build cost matrix (negative IoU)
            cost_matrix = np.zeros((len(self.tracks), len(boxes)))
            for i, track in enumerate(self.tracks):
                for j, box in enumerate(boxes):
                    # Only match same class
                    if track.class_id == classes[j]:
                        cost_matrix[i, j] = -iou(track.box, box)
                    else:
                        cost_matrix[i, j] = 0  # No match for different classes

            matched, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix)

            # Update matched tracks
            for match in matched:
                track_idx, det_idx = match
                if cost_matrix[track_idx, det_idx] < -self.iou_threshold:
                    self.tracks[track_idx].update(boxes[det_idx], scores[det_idx])
                else:
                    unmatched_tracks.append(track_idx)
                    unmatched_dets.append(det_idx)
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(boxes)))

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = TrackedObject(boxes[det_idx], classes[det_idx], scores[det_idx])
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Prepare output (only confirmed tracks)
        result_boxes = []
        result_classes = []
        result_scores = []
        result_ids = []

        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                result_boxes.append(track.box)
                result_classes.append(track.class_id)
                result_scores.append(track.score)
                result_ids.append(track.id)

        return {
            "boxes": result_boxes,
            "classes": result_classes,
            "scores": result_scores,
            "track_ids": result_ids,
        }

    def reset(self):
        """Reset all tracks."""
        self.tracks = []
        self.frame_count = 0
        TrackedObject.reset_id_counter()
