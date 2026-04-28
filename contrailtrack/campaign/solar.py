"""Sunrise/sunset calculation and daylight window generation.

Uses the standard NOAA solar-position algorithm (accurate to ~1 minute)
to avoid an external dependency on ``astral`` or ``ephem``.
"""

from __future__ import annotations

import datetime as dt
import math


def sunrise_sunset(
    date: dt.date, lat: float, lon: float
) -> tuple[dt.datetime, dt.datetime]:
    """Compute sunrise and sunset (UTC) for a given date and location.

    Based on the NOAA Solar Calculator spreadsheet formulas.
    Accuracy: within ~1 minute for mid-latitudes.
    """
    # Julian day number
    a = (14 - date.month) // 12
    y = date.year + 4800 - a
    m = date.month + 12 * a - 3
    jdn = date.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn - 0.5  # Julian date at 0h UT

    # Julian century from J2000.0
    n = jd - 2451545.0
    jc = n / 36525.0

    # Solar mean anomaly (degrees)
    M = (357.5291092 + 35999.0502909 * jc) % 360

    # Equation of center (degrees)
    Mr = math.radians(M)
    C = (1.9146 - 0.004817 * jc) * math.sin(Mr) + 0.019993 * math.sin(2 * Mr) + 0.00029 * math.sin(3 * Mr)

    # Sun's ecliptic longitude
    omega = (125.04 - 1934.136 * jc) % 360
    sun_lon = (M + C + 180.0 + 102.93735) % 360
    apparent_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Obliquity of ecliptic
    obliquity = 23.439291 - 0.0130042 * jc + 0.00256 * math.cos(math.radians(omega))
    obliquity_r = math.radians(obliquity)

    # Solar declination
    decl = math.asin(math.sin(obliquity_r) * math.sin(math.radians(apparent_lon)))

    # Equation of time (minutes)
    e = obliquity_r / 2
    y_eot = math.tan(e) ** 2
    l0 = math.radians((280.46646 + 36000.76983 * jc) % 360)
    eot = 4.0 * math.degrees(
        y_eot * math.sin(2 * l0)
        - 2.0 * math.sin(Mr) * (1 - 2 * y_eot * math.cos(2 * l0))  # keep sign
        + 4.0 * math.sin(Mr) * y_eot * math.cos(2 * l0)  # correction
        - 0.5 * y_eot ** 2 * math.sin(4 * l0)
        - 1.25 * math.sin(Mr) ** 2
    )

    # Simpler, more robust EoT from the NOAA spreadsheet:
    # Recompute using the direct formula.
    B = 2 * math.pi * (n - 81) / 365
    eot = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    # Hour angle at sunrise/sunset (degrees)
    lat_r = math.radians(lat)
    # Standard refraction correction: -0.833 degrees
    cos_ha = (
        math.cos(math.radians(90.833)) - math.sin(lat_r) * math.sin(decl)
    ) / (math.cos(lat_r) * math.cos(decl))

    cos_ha = max(-1.0, min(1.0, cos_ha))
    ha = math.degrees(math.acos(cos_ha))

    # Solar noon in minutes from midnight UTC
    solar_noon_min = 720.0 - 4.0 * lon - eot

    sunrise_min = solar_noon_min - ha * 4.0
    sunset_min = solar_noon_min + ha * 4.0

    base = dt.datetime(date.year, date.month, date.day)
    rise = base + dt.timedelta(minutes=sunrise_min)
    sset = base + dt.timedelta(minutes=sunset_min)

    return rise, sset


def daylight_windows(
    date: dt.date,
    lat: float,
    lon: float,
    window_hours: float = 2.0,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Split daylight hours into non-overlapping windows of fixed duration.

    The last window may be shorter than ``window_hours`` if it would extend
    past sunset.
    """
    rise, sset = sunrise_sunset(date, lat, lon)

    # Round sunrise up to the nearest 30-second boundary (image cadence)
    rise = _ceil_30s(rise)
    # Round sunset down to nearest 30-second boundary
    sset = _floor_30s(sset)

    if sset <= rise:
        return []

    delta = dt.timedelta(hours=window_hours)
    windows: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = rise

    while cursor < sset:
        end = min(cursor + delta, sset)
        windows.append((cursor, end))
        cursor = end

    return windows


def _ceil_30s(t: dt.datetime) -> dt.datetime:
    """Round a datetime up to the next 30-second boundary."""
    s = t.second
    remainder = s % 30
    if remainder == 0 and t.microsecond == 0:
        return t
    add = 30 - remainder if remainder else 30
    return t.replace(microsecond=0) + dt.timedelta(seconds=add)


def _floor_30s(t: dt.datetime) -> dt.datetime:
    """Round a datetime down to the previous 30-second boundary."""
    return t.replace(second=t.second - t.second % 30, microsecond=0)
