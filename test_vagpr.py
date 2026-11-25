from dataclasses import replace
from datetime import date, datetime, timedelta
import tempfile
import unittest
from pathlib import Path

import vagpr
import scan
import numpy as np
from numpy.polynomial.hermite import hermgauss


class VAGPRTest(unittest.TestCase):
    @staticmethod
    def bars():
        return [
            vagpr.Bar((date(2024, 1, 1)+timedelta(days=i)).isoformat(),
                      p, p, 1.01*p, .99*p, 1000+i%20, p)
            for i in range(220)
            for p in [12*np.exp(.002*i + .03*np.sin(i/4))]
        ]

    def test_conditioned_kernel_zero_integral_and_psd(self):
        nodes, weights = hermgauss(48)
        x = np.array([-2., -.2, 1.5])
        k = vagpr.centered_rbf(x[:, None], np.sqrt(2)*nodes)
        np.testing.assert_allclose(k @ weights / np.sqrt(np.pi), 0, atol=1e-11)
        points = np.random.default_rng(0).normal(size=(18, 6))
        cov = vagpr.covariance(vagpr.scalar_orders(points, points), [-1, -2, .8, -.4, 0])
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), -1e-10)

    def test_causal_training_and_features(self):
        bars = self.bars()
        later = [replace(b, open=b.open*1.2, close=b.close*1.2,
                         high=b.high*1.2, low=b.low*1.2) for b in bars[200:]]
        for h in vagpr.HORIZONS:
            original = vagpr.make_series(bars, h)
            revised = vagpr.make_series(bars[:200] + later, h)
            origin = 179
            train = vagpr.training_indices(origin, h)
            self.assertEqual(train.stop-1+h, origin)
            np.testing.assert_array_equal(original.x[:origin+1], revised.x[:origin+1])
            np.testing.assert_array_equal(original.y[train], revised.y[train])
            self.assertNotEqual(original.y[origin, 0], revised.y[origin, 0])

    def test_csv_roundtrip_and_bad_rows(self):
        bars = self.bars()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bars.csv"
            vagpr.save_csv(path, bars)
            self.assertEqual(bars, vagpr.load_csv(path))
            with self.assertRaises(ValueError):
                vagpr.validate_bars(bars + [bars[-1]])
            with self.assertRaises(ValueError):
                vagpr.validate_bars(bars[:-1] + [replace(bars[-1], close=np.nan)])

    def test_session_cutoff(self):
        self.assertEqual(vagpr.last_complete_day(datetime(2026, 11, 25, 14, 59, tzinfo=vagpr.CN)),
                         date(2026, 11, 24))
        self.assertEqual(vagpr.last_complete_day(datetime(2026, 11, 25, 15, 15, tzinfo=vagpr.CN)),
                         date(2026, 11, 25))

    def test_forecast_and_intervals_without_network(self):
        result = vagpr.run(self.bars(), 10)
        self.assertEqual(result["as_of"], self.bars()[-1].date)
        self.assertLess(result["return_lower"], result["return"])
        self.assertLess(result["return"], result["return_upper"])
        self.assertFalse(result["calibration"]["ready"])
        self.assertEqual(result["return"], 0)
        self.assertIsNone(result["backtest"]["model"])
        self.assertFalse(result["backtest"]["passes_screen"])

    def test_return_calibration_prefers_sensible_baseline(self):
        predicted = np.full(60, .05)
        sigma = predicted.copy()
        poor = vagpr.calibrate_returns(predicted, sigma, -predicted, .10, .05)
        self.assertTrue(poor.ready)
        self.assertEqual(poor.weight, 0)
        self.assertEqual(poor.center, 0)
        good = vagpr.calibrate_returns(predicted, sigma, predicted, .10, .05)
        self.assertEqual(good.weight, 1)
        self.assertAlmostEqual(good.center, .10)
        self.assertEqual(good.n, vagpr.CAL_WINDOW)

    def test_return_calibration_never_uses_unsettled_targets(self):
        rng = np.random.default_rng(21)
        actual = rng.normal(0, .04, 65)
        predicted = actual+rng.normal(0, .03, 65)
        sigma = np.linspace(.03, .06, 65)
        full = vagpr.calibrate_returns(predicted, sigma, actual, .10, .04)
        for stop in (5, 20, 55):
            replay = vagpr.calibrate_returns(predicted[:stop], sigma[:stop], actual[:stop],
                                             predicted[stop], sigma[stop])
            self.assertEqual(replay.center, full.past_centers[stop])
            self.assertEqual(replay.width, full.past_widths[stop])

    def test_price_and_trading_date_ticks(self):
        for series in ([13.52, 14.47], [13.52]*60, [.15, .21], [122, 1900]):
            ticks = vagpr.price_ticks(series, 5)
            self.assertLessEqual(ticks[0], min(series))
            self.assertGreaterEqual(ticks[-1], max(series))
            np.testing.assert_allclose(np.diff(ticks), ticks[1]-ticks[0])
        indices = vagpr.date_ticks(250, 700)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 249)
        self.assertEqual(indices, sorted(set(indices)))

    def test_market_symbols_and_pe_boundaries(self):
        self.assertEqual(vagpr.market_symbol("000001"), "sz000001")
        self.assertEqual(vagpr.market_symbol("688009"), "sh688009")
        with self.assertRaises(ValueError):
            vagpr.market_symbol("123456")
        members = {str(i).zfill(6): f"S{i}" for i in range(7)}
        snap = {code: ("20261125161500", pe, "13") for code, pe in zip(
            members, ("0", "20", "19.99", "-2", "not available"))}
        snap["000005"] = ("20261124161500", "15", "13")
        session, rows, exclusions = scan.select_quotes(
            members, snap, datetime(2026, 11, 25, 17, 10, tzinfo=vagpr.CN))
        self.assertEqual(session, "2026-11-25")
        self.assertEqual([r[0] for r in rows], ["000002"])
        self.assertEqual(exclusions, dict(missing_quote=1, missing_pe_or_price=1, different_session=1))


if __name__ == "__main__":
    unittest.main()
