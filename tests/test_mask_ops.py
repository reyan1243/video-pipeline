import unittest

import numpy as np
from scipy import ndimage

from mask_ops import clean_mask, clean_masks, feather_mask, is_disrupted, mask_iou, temporal_median
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


def _square(size: int = 100, half: int = 20) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    centre = size // 2
    mask[centre - half : centre + half, centre - half : centre + half] = 255
    return mask


class TestFeatherMask(unittest.TestCase):
    def test_produces_intermediate_alpha(self):
        # The whole point: a binary silhouette has a stair-stepped edge. A matte
        # needs values between 0 and 255 along the boundary.
        feathered = feather_mask(_square(), sigma=1.0)
        intermediate = np.count_nonzero((feathered > 0) & (feathered < 255))
        self.assertGreater(intermediate, 0)

    def test_interior_and_exterior_stay_saturated(self):
        feathered = feather_mask(_square(), sigma=1.0)
        self.assertEqual(feathered[50, 50], 255)  # deep inside
        self.assertEqual(feathered[5, 5], 0)  # far outside

    def test_zero_sigma_is_a_no_op(self):
        mask = _square()
        np.testing.assert_array_equal(feather_mask(mask, sigma=0), mask)

    def test_edge_is_biased_inward(self):
        # Erode-then-blur, not blur alone: text tucked a pixel under a shoulder
        # reads as correct, text creeping a pixel over it reads as a bug.
        mask = _square()
        feathered = feather_mask(mask, sigma=1.0)
        self.assertLess(int(feathered.sum()), int(mask.sum()))

    def test_clean_mask_feathers_by_default(self):
        cleaned = clean_mask(_square())
        self.assertTrue(np.any((cleaned > 0) & (cleaned < 255)))


class TestCleanMasksInPlace(unittest.TestCase):
    def test_mutates_the_same_dict(self):
        # A dict comprehension held the old and new dicts simultaneously, doubling
        # peak RAM (3.2 GB vs 1.6 GB on a minute of 720p) at the worst moment.
        masks = {0: _square(), 1: _square()}
        returned = clean_masks(masks)
        self.assertIs(returned, masks)

    def test_replaces_every_entry(self):
        masks = {0: _square(), 1: _square()}
        originals = {idx: mask.copy() for idx, mask in masks.items()}
        clean_masks(masks)
        for idx in originals:
            self.assertFalse(np.array_equal(masks[idx], originals[idx]))

    def test_empty_masks_survive(self):
        masks = {0: np.zeros((10, 10), dtype=np.uint8)}
        clean_masks(masks)
        self.assertEqual(int(masks[0].sum()), 0)


class TestTemporalMedian(unittest.TestCase):
    @staticmethod
    def _flat(value: int) -> np.ndarray:
        return np.full((8, 8), value, dtype=np.uint8)

    def test_removes_a_single_frame_spike(self):
        masks = {0: self._flat(0), 1: self._flat(255), 2: self._flat(0)}
        temporal_median(masks)
        self.assertEqual(int(masks[1].max()), 0)

    def test_leaves_a_stable_sequence_untouched(self):
        masks = {idx: self._flat(255) for idx in range(5)}
        temporal_median(masks)
        for mask in masks.values():
            self.assertEqual(int(mask.min()), 255)

    def test_does_not_feed_smoothed_output_back_in(self):
        # Each window must be built from pre-smoothing originals; otherwise the
        # filter cascades and a sustained edge erodes frame by frame.
        masks = {0: self._flat(0), 1: self._flat(255), 2: self._flat(255), 3: self._flat(255)}
        temporal_median(masks)
        self.assertEqual(int(masks[2].min()), 255)
        self.assertEqual(int(masks[3].min()), 255)

    def test_too_short_to_smooth_is_returned_unchanged(self):
        masks = {0: self._flat(0), 1: self._flat(255)}
        temporal_median(masks)
        self.assertEqual(int(masks[1].min()), 255)


if __name__ == "__main__":
    unittest.main()
