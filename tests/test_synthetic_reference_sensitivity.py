"""Input separation and support honesty for the prior-release diagnostic."""

from copy import deepcopy
import unittest

from tools.synthetic_sync.biased_references import biased_reference_case
from tools.synthetic_sync.reference_sensitivity import (
    measure_sensitivity,
    released_request,
    stored_anchor_prior_pick_errors,
)


class ReferenceSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.case = biased_reference_case("biased-hard")
        self.request = self.case["request"]
        camera = self.request["cameras"][0]
        point = self.request["points"][0]
        self.baseline = dict(success=True, cameras={camera["id"]: deepcopy(camera)},
                             landmarks={point["id"]: deepcopy(self.case["truth"]["points"][point["id"]])})

    def test_release_changes_only_point_priors(self):
        original = deepcopy(self.request)
        released = released_request(self.request)
        self.assertEqual(self.request, original)
        self.assertEqual(len([point for point in self.request["points"] if point["known"] is not None]), 4)
        self.assertTrue(all(point["known"] is None for point in released["points"]))
        for before, after in zip(self.request["points"], released["points"]):
            self.assertEqual({key: value for key, value in before.items() if key != "known"},
                             {key: value for key, value in after.items() if key != "known"})
        for key in self.request.keys() - {"points"}:
            self.assertEqual(self.request[key], released[key])

    def test_truth_and_condition_metadata_cannot_change_signal(self):
        released = deepcopy(self.baseline)
        first = measure_sensitivity(self.case["request"], self.baseline, released)
        altered = deepcopy(self.case)
        altered["truth"] = {"points": "poison"}
        altered["expectation"] = {"outcome": "poison"}
        altered["reference_experiment"] = {"condition": "reversed", "biased_ids": ["wrong"]}
        altered["name"] = "poison"
        second = measure_sensitivity(altered["request"], self.baseline, released)
        self.assertEqual(first, second)

    def test_existing_anchor_guard_numerical_counterpart(self):
        for noise_px in (0.0, 0.3):
            accurate = biased_reference_case("unbiased-hard", noise_px=noise_px)
            biased = biased_reference_case("biased-hard", noise_px=noise_px)
            accurate_errors = stored_anchor_prior_pick_errors(accurate["request"])
            biased_errors = stored_anchor_prior_pick_errors(biased["request"])
            self.assertEqual(len(accurate_errors), 4)
            self.assertEqual(len(biased_errors), 4)
            self.assertFalse([key for key, value in accurate_errors.items() if value is None or value > 5.0])
            self.assertEqual(len([key for key, value in biased_errors.items()
                                  if value is None or value > 5.0]), 2)

    def test_missing_solved_support_is_explicit_and_not_comparable(self):
        released = deepcopy(self.baseline)
        released["cameras"].clear()
        released["landmarks"].clear()
        signal = measure_sensitivity(self.request, self.baseline, released)
        self.assertFalse(signal["comparable"])
        self.assertEqual(signal["support_loss"]["cameras"], ["view_0"])
        self.assertEqual(len(signal["support_loss"]["landmarks"]), 1)
        self.assertIsNone(signal["fit"].get("all", {}).get("released_rms_px"))

    def test_equal_partial_results_are_not_called_comparable(self):
        partial = deepcopy(self.baseline)
        signal = measure_sensitivity(self.request, partial, deepcopy(partial))
        self.assertFalse(signal["comparable"])
        self.assertEqual(len(signal["support_loss"]["missing_references"]["baseline"]), 3)
        self.assertEqual(signal["support_loss"]["missing_references"]["baseline"],
                         signal["support_loss"]["missing_references"]["released"])


if __name__ == "__main__":
    unittest.main()
