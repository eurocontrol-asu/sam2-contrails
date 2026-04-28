"""Tests for contrailtrack.campaign.solar — sunrise/sunset and daylight windows."""

import datetime as dt
import unittest


class TestSunriseSunset(unittest.TestCase):
    """Verify sunrise/sunset for Paris (Brétigny) against known values."""

    PARIS_LAT = 48.600518
    PARIS_LON = 2.346795

    def test_winter_solstice_area(self):
        """Dec 31 2025 Paris: sunrise ~07:45-07:55 UTC, sunset ~16:00-16:10 UTC."""
        from contrailtrack.campaign.solar import sunrise_sunset

        date = dt.date(2025, 12, 31)
        rise, sset = sunrise_sunset(date, self.PARIS_LAT, self.PARIS_LON)

        self.assertIsInstance(rise, dt.datetime)
        self.assertIsInstance(sset, dt.datetime)
        # Sunrise between 07:40 and 08:00 UTC
        self.assertGreaterEqual(rise.hour, 7)
        self.assertLessEqual(rise, dt.datetime(2025, 12, 31, 8, 0))
        # Sunset between 15:55 and 16:15 UTC
        self.assertGreaterEqual(sset, dt.datetime(2025, 12, 31, 15, 55))
        self.assertLessEqual(sset, dt.datetime(2025, 12, 31, 16, 15))

    def test_summer_long_days(self):
        """Jun 21 2025 Paris: sunrise ~03:45-04:00 UTC, sunset ~19:50-20:10 UTC."""
        from contrailtrack.campaign.solar import sunrise_sunset

        date = dt.date(2025, 6, 21)
        rise, sset = sunrise_sunset(date, self.PARIS_LAT, self.PARIS_LON)

        # Sunrise before 04:05 UTC
        self.assertLess(rise, dt.datetime(2025, 6, 21, 4, 5))
        # Sunset after 19:45 UTC
        self.assertGreater(sset, dt.datetime(2025, 6, 21, 19, 45))

    def test_equinox(self):
        """Mar 20 2025 Paris: sunrise ~05:50-06:10 UTC, sunset ~17:50-18:10 UTC."""
        from contrailtrack.campaign.solar import sunrise_sunset

        date = dt.date(2025, 3, 20)
        rise, sset = sunrise_sunset(date, self.PARIS_LAT, self.PARIS_LON)

        self.assertGreater(rise, dt.datetime(2025, 3, 20, 5, 40))
        self.assertLess(rise, dt.datetime(2025, 3, 20, 6, 15))
        self.assertGreater(sset, dt.datetime(2025, 3, 20, 17, 45))
        self.assertLess(sset, dt.datetime(2025, 3, 20, 18, 15))


class TestDaylightWindows(unittest.TestCase):
    """Verify 2-hour window splitting."""

    PARIS_LAT = 48.600518
    PARIS_LON = 2.346795

    def test_winter_windows(self):
        """Dec 31 2025: ~8h daylight → 4 windows, last may be shorter."""
        from contrailtrack.campaign.solar import daylight_windows

        windows = daylight_windows(dt.date(2025, 12, 31), self.PARIS_LAT, self.PARIS_LON)

        self.assertGreaterEqual(len(windows), 3)
        self.assertLessEqual(len(windows), 5)

        for start, stop in windows:
            self.assertIsInstance(start, dt.datetime)
            self.assertIsInstance(stop, dt.datetime)
            self.assertGreater(stop, start)
            # Each window ≤ 2 hours
            self.assertLessEqual((stop - start).total_seconds(), 2 * 3600 + 1)

        # Windows are contiguous and non-overlapping
        for i in range(len(windows) - 1):
            self.assertEqual(windows[i][1], windows[i + 1][0])

    def test_summer_more_windows(self):
        """Jun 21 2025: ~16h daylight → 7-8 windows."""
        from contrailtrack.campaign.solar import daylight_windows

        windows = daylight_windows(dt.date(2025, 6, 21), self.PARIS_LAT, self.PARIS_LON)

        self.assertGreaterEqual(len(windows), 7)
        self.assertLessEqual(len(windows), 9)

    def test_custom_window_size(self):
        """3-hour windows should produce fewer windows."""
        from contrailtrack.campaign.solar import daylight_windows

        w2 = daylight_windows(dt.date(2025, 6, 21), self.PARIS_LAT, self.PARIS_LON, window_hours=2.0)
        w3 = daylight_windows(dt.date(2025, 6, 21), self.PARIS_LAT, self.PARIS_LON, window_hours=3.0)

        self.assertGreater(len(w2), len(w3))

    def test_first_window_starts_at_sunrise(self):
        """First window should start at (or just after) sunrise."""
        from contrailtrack.campaign.solar import sunrise_sunset, daylight_windows

        date = dt.date(2025, 12, 31)
        rise, _ = sunrise_sunset(date, self.PARIS_LAT, self.PARIS_LON)
        windows = daylight_windows(date, self.PARIS_LAT, self.PARIS_LON)

        # First window starts within 1 minute of sunrise
        delta = abs((windows[0][0] - rise).total_seconds())
        self.assertLessEqual(delta, 60)

    def test_last_window_ends_at_sunset(self):
        """Last window should end at (or just before) sunset."""
        from contrailtrack.campaign.solar import sunrise_sunset, daylight_windows

        date = dt.date(2025, 12, 31)
        _, sset = sunrise_sunset(date, self.PARIS_LAT, self.PARIS_LON)
        windows = daylight_windows(date, self.PARIS_LAT, self.PARIS_LON)

        delta = abs((windows[-1][1] - sset).total_seconds())
        self.assertLessEqual(delta, 60)


if __name__ == "__main__":
    unittest.main()
