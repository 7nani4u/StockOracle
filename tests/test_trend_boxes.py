import math
import unittest

from market_briefing.trend_boxes import analyze_trend_boxes


def bars(values):
    return {"opens": list(values), "closes": list(values),
            "highs": [v+.6 for v in values], "lows": [v-.6 for v in values],
            "volumes": [1000]*len(values)}


class TrendBoxTests(unittest.TestCase):
    def test_directions(self):
        for slope, expected in ((.8, "ascending"), (-.8, "descending"), (0, "flat")):
            r = analyze_trend_boxes(bars([200+slope*i+.2*math.sin(i) for i in range(122)]))
            self.assertEqual(r["scales"]["short"]["current"]["direction"], expected)
            self.assertFalse(r["entry_zone"]["eligible"])

    def test_upward_reversal_and_no_unfounded_targets(self):
        values = [120-.4*i for i in range(20)] + [112+.5*i for i in range(20)] + [122, 122.5]
        r = analyze_trend_boxes(bars(values))
        self.assertEqual(r["transition"]["state"], "upward_reversal")
        self.assertNotIn("targets", r)
        self.assertFalse(r["entry_zone"]["eligible"], "An extended breakout must wait for retest")

    def test_downward_reversal(self):
        values = [100+.3*i for i in range(20)] + [105-.5*i for i in range(20)] + [94, 93]
        r = analyze_trend_boxes(bars(values))
        self.assertEqual(r["transition"]["state"], "failed_upward_reversal")
        values = [100]*20 + [100-.5*i for i in range(20)] + [89, 88]
        self.assertEqual(analyze_trend_boxes(bars(values))["transition"]["state"], "downward_reversal")

    def test_retest_enables_entry(self):
        values = [104-.2*i for i in range(20)] + [99+.11*i for i in range(20)] + [101.2, 101.3]
        r = analyze_trend_boxes(bars(values))
        self.assertTrue(r["entry_zone"]["eligible"])
        self.assertLessEqual(r["entry_zone"]["low"], r["entry_zone"]["high"])
        self.assertIsNone(r["scales"]["long"]["current"])
        self.assertEqual(r["scales"]["long"]["transition"]["state"], "insufficient")

    def test_long_decline_blocks_short_recovery(self):
        values = ([180-.8*i for i in range(80)] + [104-.2*i for i in range(20)]
                  + [99+.11*i for i in range(20)] + [101.2, 101.3])
        r = analyze_trend_boxes(bars(values))
        self.assertTrue(r["scale_conflict"])
        self.assertFalse(r["entry_zone"]["eligible"])
        self.assertFalse(r["transition"]["eligible"])

    def test_requested_windows_never_silently_shrink(self):
        r = analyze_trend_boxes(bars([100]*30))
        self.assertEqual(r["status"], "insufficient_data")
        self.assertIsNone(r["scales"]["short"]["current"])

    def test_nonpositive_projected_channel_is_withheld(self):
        values = [100-4*i for i in range(20)] + [20]*20 + [20, 20]
        r = analyze_trend_boxes(bars(values))
        self.assertIsNone(r["scales"]["short"]["current"])
        self.assertFalse(r["entry_zone"]["eligible"])

    def test_recovery_support_failure(self):
        values = [120-.4*i for i in range(20)] + [112+.5*i for i in range(20)] + [108, 107]
        self.assertEqual(analyze_trend_boxes(bars(values))["transition"]["state"], "failed_upward_reversal")

    def test_prefix_causality_and_confirmation_is_withheld(self):
        values = [200+.4*i+.2*math.sin(i) for i in range(100)]
        full = bars(values)
        prefix = {k: v[:80] for k, v in full.items()}
        before = analyze_trend_boxes(prefix)
        full["closes"][90] = 1000
        self.assertEqual(before, analyze_trend_boxes({k: v[:80] for k, v in full.items()}))
        changed = bars(values[:78]+[500, 501])
        after = analyze_trend_boxes(changed)
        self.assertEqual(before["scales"]["short"]["current"], after["scales"]["short"]["current"])

    def test_validation(self):
        self.assertEqual(analyze_trend_boxes(bars([100]*5))["status"], "insufficient_data")
        bad = bars([100]*50)
        bad["lows"][2] = 101
        self.assertEqual(analyze_trend_boxes(bad)["status"], "invalid_data")
        bad = bars([100]*50)
        bad["dates"] = ["2026-01-01"]*50
        self.assertEqual(analyze_trend_boxes(bad)["status"], "invalid_data")
        for config in ({"short_window": 2}, {"confirmation_bars": 1}, {"entry_band_atr": float("nan")}, {"long_window": 2}):
            with self.assertRaises(ValueError):
                analyze_trend_boxes(bars([100]*50), config)


if __name__ == "__main__":
    unittest.main()
