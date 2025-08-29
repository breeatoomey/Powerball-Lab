import numpy as np
import pandas as pd

def add_calendar_features(df):
    dts = df["draw_date"].dt.tz_convert("US/Eastern")
    dow = dts.dt.weekday
    month = dts.dt.month
    doy = dts.dt.dayofyear

    month_sin = np.sin(2*np.pi*(month/12.0))
    month_cos = np.cos(2*np.pi*(month/12.0))
    doy_sin   = np.sin(2*np.pi*(doy/366.0))
    doy_cos   = np.cos(2*np.pi*(doy/366.0))

    out = pd.DataFrame({
        "dow": dow.astype(int),
        "month": month.astype(int),
        "doy": doy.astype(int),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
    }, index=df.index)
    return out

def build_number_targets(df, white_max=69, pb_max=26):
    whites = pd.DataFrame(0, index=df.index, columns=range(1, white_max+1))
    for i in range(1,6):
        whites = whites.add(pd.get_dummies(df[f"w{i}"], columns=range(1, white_max+1)), fill_value=0)
    whites = (whites > 0).astype(int)
    pb = pd.get_dummies(df["pb"]).reindex(columns=range(1, pb_max+1), fill_value=0)
    return whites, pb
