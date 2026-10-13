import math
import unittest

from market_briefing.box_range import analyze_box_range


def series(tail=(100, 100, 100), scale=1):
    closes = [100 + 5*math.sin(i*math.pi/4) for i in range(64)] + list(tail)
    return {"opens": [c*scale for c in closes], "closes": [c*scale for c in closes],
            "highs": [(c+.5)*scale for c in closes],
            "lows": [(c-.5)*scale for c in closes], "volumes": [1000]*len(closes)}


class BoxRangeTests(unittest.TestCase):
    def test_six_market_states(self):
        cases = [((100, 100, 100), "range"), ((95, 96, 97), "support_rebound"),
                 ((105, 104, 103), "resistance_rejection"),
                 ((107, 108, 109), "breakout_confirmed"),
                 ((93, 92, 91), "breakdown_confirmed"),
                 ((108, 100, 100), "false_breakout")]
        for tail, expected in cases:
            with self.subTest(expected=expected):
                result = analyze_box_range(series(tail))
                self.assertEqual(result["state"], expected)
        trend = {k: [v + i*2 if k != "volumes" else v for i, v in enumerate(a)]
                 for k, a in series().items()}
        self.assertEqual(analyze_box_range(trend)["state"], "no_box")

    def test_entry_and_target_evidence(self):
        result = analyze_box_range(series((95, 96, 97)))
        self.assertTrue(result["entries"]["main"]["eligible"])
        self.assertNotEqual(result["entries"]["exploration"]["low"], result["entries"]["main"]["low"])
        self.assertLess(result["targets"]["conservative"]["high"], result["targets"]["neutral"]["low"])
        breakout = analyze_box_range(series((107, 108, 109)))
        self.assertFalse(breakout["targets"]["aggressive"]["eligible"])
        self.assertIsNone(breakout["targets"]["aggressive"]["high"])
        self.assertIsNone(breakout["targets"]["neutral"]["high"])
        broken = analyze_box_range(series((93, 92, 91)))
        self.assertFalse(broken["entries"]["exploration"]["eligible"])
        self.assertIsNone(broken["targets"]["conservative"]["high"])

    def test_confirmation_not_used_for_fitting(self):
        base = analyze_box_range(series())
        extreme = analyze_box_range(series((120, 125, 130)))
        self.assertEqual(base["support"], extreme["support"])
        self.assertEqual(base["resistance"], extreme["resistance"])
        full = series()
        prefix = {k: v[:60] for k, v in full.items()}
        before = analyze_box_range(prefix)
        full["closes"][-1] = 999
        self.assertEqual(before, analyze_box_range({k: v[:60] for k, v in full.items()}))

    def test_scale_and_data_validation(self):
        first, scaled = analyze_box_range(series()), analyze_box_range(series(scale=100))
        self.assertEqual(first["state"], scaled["state"])
        self.assertAlmostEqual(first["support"]["low"]*100, scaled["support"]["low"])
        for bad in (float("nan"), -1, float("inf")):
            dd = series()
            dd["closes"][0] = bad
            self.assertEqual(analyze_box_range(dd)["state"], "insufficient_data")
        dd = series()
        dd["highs"].pop()
        self.assertEqual(analyze_box_range(dd)["state"], "insufficient_data")

    def test_pending_and_historical_aggressive_resistance(self):
        self.assertEqual(analyze_box_range(series((100, 100, 110)))["state"], "breakout_pending")
        self.assertEqual(analyze_box_range(series((100, 100, 90)))["state"], "breakdown_pending")
        dd = series((107, 108, 109))
        historical = [112, 114, 116, 114, 112]*6
        for key, values in list(dd.items()):
            prefix = ([1000]*len(historical) if key == "volumes" else
                      [v+.5 for v in historical] if key == "highs" else
                      [v-.5 for v in historical] if key == "lows" else historical)
            dd[key] = prefix + values
        target = analyze_box_range(dd)["targets"]["aggressive"]
        self.assertTrue(target["eligible"])
        self.assertGreater(target["low"], dd["closes"][-1])
        self.assertLess(target["low"], 116.5)
        self.assertGreater(target["high"], 116.5)

    def test_stale_touches_and_dates(self):
        dd = series()
        for key in dd:
            dd[key] = dd[key][:40] + ([1000]*45 if key == "volumes" else
                [100.5]*45 if key == "highs" else [99.5]*45 if key == "lows" else [100]*45)
        self.assertEqual(analyze_box_range(dd)["state"], "no_box")
        dd = series()
        dd["dates"] = [f"2026-{i:03d}" for i in range(len(dd["closes"]))]
        self.assertEqual(analyze_box_range(dd)["state"], "range")
        dd["dates"][0], dd["dates"][1] = dd["dates"][1], dd["dates"][0]
        self.assertEqual(analyze_box_range(dd)["state"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()

