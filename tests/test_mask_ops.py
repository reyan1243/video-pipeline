import unittest

import numpy as np
from scipy import ndimage

import cv2

from mask_ops import (
    clean_mask,
    clean_masks,
    downscale_mask,
    feather_mask,
    fill_holes,
    is_disrupted,
    mask_iou,
    temporal_median,
)
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

    def test_keeps_side_by_side_blobs_separate(self):
        # An arm standing off the torso. Bridging these is what let fill_holes
        # flood the enclosed triangle and swallow the negative space a caption
        # behind the subject shows through, so the close must NOT join them
        # however wide the kernel is — it only ever bridges vertically.
        mask = self._dumbbell_with_gap_and_hole()
        _, before_components = ndimage.label(mask > 0)
        self.assertEqual(before_components, 2)

        cleaned = clean_mask(mask, close_kernel_size=45)
        _, after_components = ndimage.label(cleaned > 0)
        self.assertEqual(after_components, 2)

    def test_closes_horizontal_cut_across_subject(self):
        # What the close exists for: something slicing straight across the
        # subject — a burned-in graphic, a strap, a mic lead — leaving a band
        # that shows as a gap in the composite.
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[40:160, 60:140] = 255
        mask[95:105, 60:140] = 0  # 10px cut, full width of the blob
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


class TestFillHoles(unittest.TestCase):
    """fill_holes replaced scipy.ndimage.binary_fill_holes for speed (59ms ->
    3.6ms per 1080x1920 frame). These pin that it is a true drop-in."""

    def _cases(self):
        simple = np.zeros((120, 120), np.uint8)
        simple[20:100, 20:100] = 255
        simple[50:70, 50:70] = 0  # interior hole

        touching = np.zeros((120, 120), np.uint8)
        touching[0:60, 0:60] = 255  # subject touching the (0,0) corner
        touching[20:30, 20:30] = 0

        two_blobs = np.zeros((120, 120), np.uint8)
        two_blobs[10:50, 10:50] = 255
        two_blobs[70:110, 70:110] = 255
        two_blobs[20:30, 20:30] = 0

        notched = np.zeros((120, 120), np.uint8)
        notched[20:100, 20:100] = 255
        notched[50:70, 20:60] = 0  # open notch — NOT a hole, must stay open

        return {
            "simple": simple,
            "corner_touching": touching,
            "two_blobs": two_blobs,
            "open_notch": notched,
            "empty": np.zeros((40, 40), np.uint8),
            "full": np.full((40, 40), 255, np.uint8),
        }

    def test_matches_scipy_exactly(self):
        for name, mask in self._cases().items():
            with self.subTest(case=name):
                expected = ndimage.binary_fill_holes(mask > 0).astype(np.uint8) * 255
                np.testing.assert_array_equal(fill_holes(mask), expected)

    def test_fills_an_interior_hole(self):
        mask = self._cases()["simple"]
        self.assertEqual(mask[60, 60], 0)
        self.assertEqual(fill_holes(mask)[60, 60], 255)

    def test_leaves_an_open_notch_open(self):
        # A notch reaching the background is not enclosed, so filling it would
        # swallow negative space a caption is meant to show through.
        filled = fill_holes(self._cases()["open_notch"])
        self.assertEqual(filled[60, 30], 0)

    def test_survives_subject_touching_the_corner(self):
        # Flooding from the raw (0,0) pixel would fail here; the 1px zero border
        # is what makes the seed provably background.
        filled = fill_holes(self._cases()["corner_touching"])
        self.assertEqual(filled[25, 25], 255)
        self.assertEqual(filled[100, 100], 0)

    def test_preserves_shape_and_dtype(self):
        mask = self._cases()["simple"]
        out = fill_holes(mask)
        self.assertEqual(out.shape, mask.shape)
        self.assertEqual(out.dtype, np.uint8)


class TestDownscaleMask(unittest.TestCase):
    def test_caps_height_and_preserves_aspect(self):
        mask = np.zeros((1920, 1080), np.uint8)
        out = downscale_mask(mask, 960)
        self.assertEqual(out.shape, (960, 540))

    def test_no_op_when_already_small_enough(self):
        mask = np.zeros((480, 270), np.uint8)
        self.assertIs(downscale_mask(mask, 960), mask)

    def test_zero_disables(self):
        mask = np.zeros((1920, 1080), np.uint8)
        self.assertIs(downscale_mask(mask, 0), mask)

    def test_keeps_the_silhouette(self):
        mask = np.zeros((1920, 1080), np.uint8)
        cv2.circle(mask, (540, 960), 300, 255, -1)
        out = downscale_mask(mask, 960)
        self.assertEqual(int(out[480, 270]), 255)  # centre still solid
        self.assertEqual(int(out[10, 10]), 0)  # corner still empty


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
