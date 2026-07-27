import unittest

import numpy as np

from datatypes import BoundingBox, TrackerConfig, TrackingDiagnostic, TrackingResult


class TestBoundingBox(unittest.TestCase):
    def test_area(self):
        self.assertEqual(BoundingBox(0, 0, 10, 20).area, 200)

    def test_area_clamps_degenerate_coords_to_zero(self):
        self.assertEqual(BoundingBox(10, 10, 5, 5).area, 0.0)

    def test_as_list_and_from_list_round_trip(self):
        box = BoundingBox(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(BoundingBox.from_list(box.as_list()), box)


class TestTrackingResult(unittest.TestCase):
    def test_frame_indices_sorted(self):
        result = TrackingResult(masks={5: np.zeros((2, 2)), 1: np.zeros((2, 2)), 3: np.zeros((2, 2))})
        self.assertEqual(result.frame_indices(), [1, 3, 5])

    def test_coverage(self):
        result = TrackingResult(masks={0: np.zeros((2, 2)), 1: np.zeros((2, 2))})
        self.assertEqual(result.coverage(4), 0.5)

    def test_coverage_zero_total_frames(self):
        result = TrackingResult(masks={})
        self.assertEqual(result.coverage(0), 0.0)

    def test_diagnostics_default_empty(self):
        result = TrackingResult(masks={})
        self.assertEqual(result.diagnostics, [])

    def test_diagnostic_fields(self):
        diag = TrackingDiagnostic(frame_index=3, direction="forward", iou=0.9, area_ratio=1.02, object_score=1.5)
        self.assertEqual(diag.frame_index, 3)
        self.assertIsNone(TrackingDiagnostic(0, "backward", 1.0, 1.0, None).object_score)


class TestTrackerConfig(unittest.TestCase):
    def test_defaults_match_documented_thresholds(self):
        config = TrackerConfig()
        self.assertEqual(config.object_score_threshold, 0.0)
        self.assertEqual(config.iou_drop_threshold, 0.4)
        self.assertEqual(config.area_ratio_bounds, (0.3, 1 / 0.3))
        self.assertEqual(config.max_segments_per_direction, 8)


if __name__ == "__main__":
    unittest.main()
