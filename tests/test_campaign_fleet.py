"""Tests for contrailtrack.campaign.fleet — OpenSky → Fleet JSON conversion."""

import datetime as dt
import unittest
from pathlib import Path

OPENSKY_PATH = Path("/data/contrailnet/flights/opensky/2025/2025-12-31.parquet")


@unittest.skipUnless(OPENSKY_PATH.exists(), "OpenSky parquet not available")
class TestOpenSkyToFleet(unittest.TestCase):
    """Integration tests against real OpenSky data (Dec 31 2025)."""

    def test_basic_conversion(self):
        """Convert a 2h window and get at least 1 flight."""
        from contrailtrack.campaign.fleet import opensky_to_fleet

        flights = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 10, 0, 0),
            stop=dt.datetime(2025, 12, 31, 12, 0, 0),
        )

        self.assertIsInstance(flights, list)
        self.assertGreater(len(flights), 0)

    def test_flight_dict_structure(self):
        """Each flight dict should have the flat keys pycontrails produces."""
        from contrailtrack.campaign.fleet import opensky_to_fleet

        flights = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 10, 0, 0),
            stop=dt.datetime(2025, 12, 31, 10, 30, 0),
        )

        self.assertGreater(len(flights), 0)
        f = flights[0]
        # Flat dict: flight_id, longitude, latitude, time, altitude_ft
        self.assertIn("flight_id", f)
        self.assertIn("longitude", f)
        self.assertIn("latitude", f)
        self.assertIn("time", f)

    def test_altitude_filtering(self):
        """Only flights at cruise altitude (150-400 hPa) pass."""
        from contrailtrack.campaign.fleet import opensky_to_fleet

        import numpy as np
        from pycontrails.physics import units

        flights = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 10, 0, 0),
            stop=dt.datetime(2025, 12, 31, 10, 30, 0),
            level_min_hpa=150.0,
            level_max_hpa=400.0,
        )

        for f in flights:
            alt_ft = np.array(f["altitude_ft"])
            levels = units.ft_to_pl(alt_ft)
            self.assertTrue(np.all(levels >= 150.0))
            self.assertTrue(np.all(levels < 400.0))

    def test_time_filtering(self):
        """Waypoints are within the time window (±1 resample interval tolerance)."""
        from contrailtrack.campaign.fleet import opensky_to_fleet
        import calendar

        start = dt.datetime(2025, 12, 31, 10, 0, 0)
        stop = dt.datetime(2025, 12, 31, 10, 30, 0)

        flights = opensky_to_fleet(OPENSKY_PATH, start=start, stop=stop)

        # to_dict() serialises time as UTC epoch seconds; use calendar.timegm (not .timestamp() which is local)
        t_start = calendar.timegm(start.timetuple()) - 15
        t_stop = calendar.timegm(stop.timetuple()) + 15
        for f in flights:
            self.assertTrue(all(t >= t_start for t in f["time"]))
            self.assertTrue(all(t <= t_stop for t in f["time"]))

    def test_narrow_window_few_flights(self):
        """A very narrow window should have fewer flights than a wide one."""
        from contrailtrack.campaign.fleet import opensky_to_fleet

        narrow = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 12, 0, 0),
            stop=dt.datetime(2025, 12, 31, 12, 1, 0),
        )
        wide = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 10, 0, 0),
            stop=dt.datetime(2025, 12, 31, 12, 0, 0),
        )

        self.assertLess(len(narrow), len(wide))

    def test_resampling(self):
        """Waypoints should be resampled to ~10s intervals."""
        from contrailtrack.campaign.fleet import opensky_to_fleet
        import numpy as np

        flights = opensky_to_fleet(
            OPENSKY_PATH,
            start=dt.datetime(2025, 12, 31, 10, 0, 0),
            stop=dt.datetime(2025, 12, 31, 10, 30, 0),
        )

        if not flights:
            self.skipTest("No flights in window")

        f = flights[0]
        times = np.array(f["time"], dtype="datetime64[ns]")
        if len(times) > 1:
            diffs = np.diff(times) / np.timedelta64(1, "s")
            self.assertTrue(np.median(diffs) <= 15.0)


if __name__ == "__main__":
    unittest.main()
