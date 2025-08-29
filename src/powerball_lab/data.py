import io
import pandas as pd
import requests
from pandas import Timestamp

WHITE_MAX = 69
PB_MAX = 26
NY_OPENDATA_CSV = "https://data.ny.gov/resource/d6yy-54nr.csv?$limit=50000"
CUTOVER = Timestamp("2015-10-07", tz="US/Eastern")  # current rules

from .utils import parse_winning_numbers

def fetch_powerball_history():
    r = requests.get(NY_OPENDATA_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["draw_date"] = pd.to_datetime(df["draw_date"]).dt.tz_localize(
        "US/Eastern", nonexistent="shift_forward", ambiguous="NaT"
    )
    # Keep only draws at/after current rules
    df = df[df["draw_date"] >= CUTOVER].copy()

    whites_list, pb_list = [], []
    for s in df["winning_numbers"].astype(str):
        w, pb = parse_winning_numbers(s)
        whites_list.append(w); pb_list.append(pb)
    for i in range(5):
        df[f"w{i+1}"] = [w[i] for w in whites_list]
    df["pb"] = pb_list
    df = df.sort_values("draw_date").reset_index(drop=True)
    return df
