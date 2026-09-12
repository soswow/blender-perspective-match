"""Prevent misleading comparisons in the line-position experiment."""

from copy import deepcopy
import unittest

from tools.synthetic_sync.line_position_gauge import compare


class GaugeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.run = {"record": {
            "request_sha256": "same-input",
            "reported_rmse_px": 0.5,
            "cameras": {"view": {"center": [0.0, 0.0, 2.0]}},
            "landmarks": {"point": [0.0, 0.0, 0.0]},
            "line_segments": {"line": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
        }}

    def test_cached_comparisons_require_identical_known_inputs(self):
        for fingerprint in (None, "different-input"):
            with self.subTest(fingerprint=fingerprint):
                other = deepcopy(self.run)
                other["record"]["request_sha256"] = fingerprint
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    compare(self.run, other)
        unknown = deepcopy(self.run)
        unknown["record"].pop("request_sha256")
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            compare(unknown, unknown)

    def test_disappearing_geometry_is_not_reported_as_zero_camera_motion(self):
        other = deepcopy(self.run)
        for field in ("cameras", "landmarks", "line_segments"):
            other["record"][field].clear()
        result = compare(self.run, other)
        self.assertIsNone(result["camera_center_max_delta"])
        for field, key in (("cameras", "view"), ("landmarks", "point"),
                           ("line_segments", "line")):
            self.assertEqual(result["support_changes"][field], {"lost": [key], "gained": []})


if __name__ == "__main__":
    unittest.main()
