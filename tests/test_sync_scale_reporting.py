"""Unknown plane membership must not be presented as absolute scale evidence."""

from copy import deepcopy
import unittest

import numpy as np

from tools.synthetic_sync.geometry import project
from tools.synthetic_sync.scenarios import generate
from tools.synthetic_sync.solver import solve


def scale_free_plane_case():
    case = generate("free_scale")
    case["request"]["plane_groups"] = [(key,"X",1) for key, point in case["truth"]["points"].items()
        if abs(point[0]-1.2) < 1e-8]
    return case


class ScaleReportingTests(unittest.TestCase):
    def test_unknown_plane_and_pixels_allow_different_absolute_sizes(self):
        case = scale_free_plane_case()
        anchor = np.asarray(case["truth"]["cameras"][0]["center"])
        points = np.asarray(list(case["truth"]["points"].values()))
        plane_members = np.asarray([case["truth"]["points"][key] for key,_,_ in case["request"]["plane_groups"]])
        self.assertGreaterEqual(len(plane_members), 2)
        for scale in (.5, 2.):
            transformed = anchor+scale*(points-anchor)
            self.assertAlmostEqual(np.linalg.norm(np.ptp(transformed,axis=0)), scale*np.linalg.norm(np.ptp(points,axis=0)))
            plane = anchor+scale*(plane_members-anchor)
            self.assertLess(np.ptp(plane[:,0]), 1e-12)
            for camera in case["truth"]["cameras"]:
                changed = deepcopy(camera)
                changed["center"] = anchor+scale*(np.asarray(camera["center"])-anchor)
                np.testing.assert_allclose(project(transformed,changed)[0], project(points,camera)[0], atol=1e-10)

    def test_unknown_plane_is_reported_as_a_constraint(self):
        result = solve(scale_free_plane_case()["request"])
        self.assertTrue(result["success"], result["message"])
        self.assertIn("constraints: 1 plane", result["message"])
        self.assertNotIn("scale from 1 plane", result["message"])


if __name__ == "__main__":
    unittest.main()
