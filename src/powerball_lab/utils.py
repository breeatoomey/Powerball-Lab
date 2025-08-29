import datetime as dt
from dateutil import tz

TZ_LOCAL = tz.gettz("America/Los_Angeles")

def today_local_date():
    return dt.datetime.now(TZ_LOCAL).date()

def next_powerball_draw_date(base_date=None):
    base_date = base_date or today_local_date()
    for offset in range(1, 8):
        d = base_date + dt.timedelta(days=offset)
        if d.weekday() in (0, 2, 5):  # Mon, Wed, Sat
            return d
    return base_date + dt.timedelta(days=2)

def parse_winning_numbers(s: str):
    parts = [int(x) for x in s.replace(",", " ").split() if x.strip()]
    whites, pb = parts[:-1], parts[-1]
    return sorted(whites), pb
