import unittest

from detector import _largest, _normalize_prompt
from datatypes import BoundingBox


class TestNormalizePrompt(unittest.TestCase):
    def test_adds_trailing_period(self):
        self.assertEqual(_normalize_prompt("person"), "person.")

    def test_leaves_trailing_period_alone(self):
        self.assertEqual(_normalize_prompt("person."), "person.")

    def test_lowercases_and_strips_whitespace(self):
        self.assertEqual(_normalize_prompt("  Guitarist  "), "guitarist.")


class TestLargest(unittest.TestCase):
    def test_picks_bigger_box_over_smaller_one(self):
        # Fabricated boxes standing in for Grounding DINO output: a small box
        # (the "guitar", if this were confidence-ranked first) and a big one
        # (the full person) — _largest must return the big one regardless of
        # any confidence ordering, since only areas are passed in here.
        small = BoundingBox(0, 0, 10, 10)
        big = BoundingBox(0, 0, 100, 200)
        self.assertEqual(_largest([small, big]), big)
        self.assertEqual(_largest([big, small]), big)

    def test_single_box(self):
        only = BoundingBox(1, 1, 5, 5)
        self.assertEqual(_largest([only]), only)


if __name__ == "__main__":
    unittest.main()
