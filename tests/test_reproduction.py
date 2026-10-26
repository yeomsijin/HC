import unittest
from pathlib import Path
import numpy as np
from hc.methods import evaluate, direction, TC_METHOD, H1_METHOD
from hc.dynamics import stationary_field, switch_field

DATA = Path(__file__).with_name("reference.npz")


class PaperRegression(unittest.TestCase):
    def test_saved_neighborhoods_costs_maps_and_predictions(self):
        with np.load(DATA) as data:
            for name in [
                "synchronized",
                "symmetry",
                "fast_slow",
                "switch16",
                "switch48",
                "atlas_odd",
                "atlas_short",
            ]:
                with self.subTest(record=name):
                    params = (
                        dict(zip(("m", "tau"), data[name + "_parameters"]))
                        if name.startswith("atlas_")
                        else {}
                    )
                    arrays, _ = evaluate(
                        data[name + "_quantiles"],
                        anchors=data[name + "_anchors"],
                        **params,
                    )
                    for key in ["neighbors_x", "neighbors_y"]:
                        np.testing.assert_array_equal(
                            arrays[key], data[name + "_" + key]
                        )
                    np.testing.assert_allclose(
                        arrays["costs"],
                        data[name + "_costs"],
                        rtol=1e-9,
                        atol=1e-11,
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        arrays["logs"],
                        data[name + "_logs"],
                        rtol=1e-7,
                        atol=1e-7,
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        arrays["ccm_predictions"],
                        data[name + "_predictions"],
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_exchange(self):
        with np.load(DATA) as data:
            q = data["fast_slow_quantiles"]
            anchors = data["fast_slow_anchors"]
            a, scores = evaluate(q, anchors=anchors)
            b, swapped = evaluate(q[:, ::-1], anchors=anchors)
            np.testing.assert_allclose(
                a["costs"][:, ::-1], b["costs"], rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(a["logs"][::-1], b["logs"], rtol=1e-8, atol=1e-9)
            for key in ["h0", "h1", "tc", "d_ccm"]:
                self.assertAlmostEqual(scores[key], -swapped[key], places=8)

    def test_zero_cost_is_undefined(self):
        self.assertTrue(np.isnan(direction(np.zeros(2))))
        self.assertTrue(np.isnan(direction(np.array([np.nan, 1.0]))))

    def test_coupling_is_outside_reciprocal_clock(self):
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        q, g = 4.0, 2.0
        d = stationary_field(z, q, g)
        self.assertEqual(d[3], 10 * (z[4] - z[3]) / q + g * (z[0] - z[3]))
        np.testing.assert_array_equal(d[:6], switch_field(z[:6], q, g, 0.0))

    def test_method_ids(self):
        self.assertEqual(TC_METHOD, "tc_main_text_source_neighbors_v1")
        self.assertEqual(
            H1_METHOD, "pce_own_own_h1_pair_coordinate_persistence_variance_v1"
        )


if __name__ == "__main__":
    unittest.main()
