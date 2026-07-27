import unittest

import numpy as np
from scipy import ndimage

from mask_ops import clean_mask, is_disrupted, mask_iou
from datatypes import TrackerConfig


class TestMaskIou(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 255
        self.assertEqual(mask_iou(mask, mask), 1.0)

    def test_disjoint_masks(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        a[0:3, 0:3] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[7:10, 7:10] = 255
        self.assertEqual(mask_iou(a, b), 0.0)

    def test_both_empty_returns_one(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        self.assertEqual(mask_iou(a, b), 1.0)


class TestCleanMask(unittest.TestCase):
    def _dumbbell_with_gap_and_hole(self) -> np.ndarray:
        # Two blobs 15px apart (smaller than the 45px close kernel) plus a small
        # interior hole punched into the first blob.
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:100, 20:60] = 255  # blob A
        mask[50:100, 75:115] = 255  # blob B, gap of 15px from A
        mask[70:76, 35:41] = 0  # interior hole inside blob A
        return mask

    def test_bridges_gap_into_one_component(self):
        mask = self._dumbbell_with_gap_and_hole()
        _, before_components = ndimage.label(mask > 0)
        self.assertEqual(before_components, 2)

        cleaned = clean_mask(mask, close_kernel_size=45)
        _, after_components = ndimage.label(cleaned > 0)
        self.assertEqual(after_components, 1)

    def test_fills_interior_hole(self):
        mask = self._dumbbell_with_gap_and_hole()
        self.assertEqual(mask[72, 37], 0)  # hole present before cleanup

        cleaned = clean_mask(mask, close_kernel_size=45)
        self.assertEqual(cleaned[72, 37], 255)  # hole filled after cleanup

    def test_empty_mask_returned_unchanged(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        cleaned = clean_mask(mask)
        np.testing.assert_array_equal(cleaned, mask)


class TestIsDisrupted(unittest.TestCase):
    def setUp(self):
        self.config = TrackerConfig()

    def test_healthy_frame_not_disrupted(self):
        self.assertFalse(is_disrupted(iou=0.9, area_ratio=1.0, object_score=1.0, config=self.config))

    def test_low_iou_is_disrupted(self):
        self.assertTrue(is_disrupted(iou=0.1, area_ratio=1.0, object_score=1.0, config=self.config))

    def test_iou_exactly_at_threshold_is_not_disrupted(self):
        # The check is a strict `<`, so the boundary value itself still passes —
        # only pins the documented semantics, not a preference.
        self.assertFalse(is_disrupted(iou=0.4, area_ratio=1.0, object_score=1.0, config=self.config))

    def test_iou_just_below_threshold_is_disrupted(self):
        self.assertTrue(is_disrupted(iou=0.399, area_ratio=1.0, object_score=1.0, config=self.config))

    def test_area_ratio_out_of_bounds_is_disrupted(self):
        self.assertTrue(is_disrupted(iou=0.9, area_ratio=0.1, object_score=1.0, config=self.config))
        self.assertTrue(is_disrupted(iou=0.9, area_ratio=5.0, object_score=1.0, config=self.config))

    def test_low_object_score_is_disrupted(self):
        self.assertTrue(is_disrupted(iou=0.9, area_ratio=1.0, object_score=-1.0, config=self.config))

    def test_object_score_none_is_ignored(self):
        self.assertFalse(is_disrupted(iou=0.9, area_ratio=1.0, object_score=None, config=self.config))


if __name__ == "__main__":
    unittest.main()
