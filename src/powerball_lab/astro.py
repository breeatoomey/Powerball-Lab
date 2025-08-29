import datetime as dt
import pandas as pd
from skyfield.api import load
from skyfield.almanac import moon_phase
from skyfield.framelib import ecliptic_frame

def compute_astro_features_for_dates(dates_local):
    ts = load.timescale()
    eph = load("de421.bsp")
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    rows = []
    for d in dates_local:
        dt_utc = dt.datetime(d.year, d.month, d.day, 3, 59, tzinfo=dt.timezone.utc)
        t = ts.from_datetime(dt_utc)
        e = earth.at(t)
        mp = moon_phase(eph, t).radians
        sun_app  = e.observe(sun).apparent()
        moon_app = e.observe(moon).apparent()
        _, sun_lon_ang,  _ = sun_app.frame_latlon(ecliptic_frame)
        _, moon_lon_ang, _ = moon_app.frame_latlon(ecliptic_frame)
        sun_lon  = float(sun_lon_ang.degrees)  % 360.0
        moon_lon = float(moon_lon_ang.degrees) % 360.0
        mercury = eph["mercury barycenter"]
        y0 = e.observe(mercury).radec()[0].hours
        t_minus = ts.from_datetime(dt_utc - dt.timedelta(days=1))
        t_plus  = ts.from_datetime(dt_utc + dt.timedelta(days=1))
        y_prev = earth.at(t_minus).observe(mercury).radec()[0].hours
        y_next = earth.at(t_plus ).observe(mercury).radec()[0].hours
        retro = int((y_next - y0) < (y0 - y_prev))
        rows.append(dict(date=d, moon_phase=mp, sun_lon=sun_lon, moon_lon=moon_lon, mercury_retro=retro))
    return pd.DataFrame(rows).set_index("date")
