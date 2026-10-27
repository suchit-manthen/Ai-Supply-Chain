"""Synthetic (but realistic) supermarket sales data generator.

Demand for each SKU = base * trend * weekday pattern * annual season * holiday lift
                      * promo lift (price elasticity) * temperature effect * noise
"""
import numpy as np
import pandas as pd

# Re-exported for backwards compatibility; the single source of truth is config.py.
from config import HORIZON, HOLDOUT, SEED

# legacy_model = the algorithm this SKU was originally hard-coded to. Kept for
# reference only; it no longer drives the forecast (see selection.py).
PRODUCTS = [
    dict(id="milk", name="Fresh Milk 1L", category="Dairy", legacy_model="ARIMA", base=240, price=1.20, cost=0.85,
         lead=2, shelf=5, stock_days=1.6, on_order=0, max_cover=6,
         weekly=[1.10, .95, .92, .95, 1.02, 1.12, 1.14], season=.03, temp=0.0, promo=.3, holiday=.05, noise=.06),
    dict(id="rice", name="Basmati Rice 5kg", category="Staples", legacy_model="ARIMA", base=95, price=9.50, cost=7.10,
         lead=4, shelf=365, stock_days=9, on_order=0, max_cover=30,
         weekly=[.85, .85, .88, .92, 1.0, 1.3, 1.25], season=.04, temp=0.0, promo=.5, holiday=.08, noise=.07),
    dict(id="icecream", name="Ice Cream Tub", category="Frozen", legacy_model="Prophet", base=70, price=4.80, cost=3.00,
         lead=3, shelf=180, stock_days=2.0, on_order=0, max_cover=14,
         weekly=[.85, .85, .9, .95, 1.1, 1.35, 1.3], season=.55, temp=.045, promo=.9, holiday=.15, noise=.10),
    dict(id="sweets", name="Festive Sweets Box", category="Seasonal", legacy_model="Prophet", base=38, price=12.0, cost=7.50,
         lead=5, shelf=60, stock_days=20, on_order=0, max_cover=35,
         weekly=[.9, .85, .9, .95, 1.05, 1.3, 1.25], season=.15, temp=0.0, promo=.8, holiday=1.8, noise=.12),
    dict(id="soda", name="Soft Drink 2L", category="Beverages", legacy_model="XGBoost", base=180, price=2.10, cost=1.30,
         lead=3, shelf=270, stock_days=2.4, on_order=0, max_cover=14,
         weekly=[.88, .85, .9, .95, 1.1, 1.3, 1.22], season=.2, temp=.03, promo=1.6, holiday=.25, noise=.08),
    dict(id="chips", name="Potato Chips 150g", category="Snacks", legacy_model="XGBoost", base=210, price=1.90, cost=1.10,
         lead=3, shelf=120, stock_days=13, on_order=0, max_cover=12,
         weekly=[.85, .82, .88, .95, 1.1, 1.35, 1.3], season=.06, temp=0.0, promo=1.4, holiday=.3, noise=.09),
    dict(id="veg", name="Fresh Vegetables (kg)", category="Produce", legacy_model="Random Forest", base=520, price=2.60, cost=1.60,
         lead=1, shelf=4, stock_days=1.1, on_order=0, max_cover=4,
         weekly=[.95, .9, .92, .95, 1.05, 1.25, 1.2], season=.10, temp=-.006, promo=.35, holiday=.12, noise=.13),
    dict(id="bread", name="Sliced Bread", category="Bakery", legacy_model="Random Forest", base=310, price=1.80, cost=1.05,
         lead=1, shelf=3, stock_days=3.8, on_order=0, max_cover=3,
         weekly=[1.05, .95, .93, .95, 1.0, 1.1, 1.08], season=.02, temp=0.0, promo=.5, holiday=.10, noise=.11),
    dict(id="eggs", name="Free Range Eggs 12 Pack", category="Dairy", legacy_model="ARIMA", base=125, price=4.20, cost=2.75,
         lead=2, shelf=28, stock_days=4.5, on_order=0, max_cover=18,
         weekly=[.92, .9, .94, .97, 1.02, 1.15, 1.1], season=.03, temp=0.0, promo=.45, holiday=.12, noise=.08),
    dict(id="yogurt", name="Greek Yogurt 500g", category="Dairy", legacy_model="Prophet", base=82, price=3.60, cost=2.20,
         lead=3, shelf=21, stock_days=2.3, on_order=0, max_cover=12,
         weekly=[.9, .88, .94, .98, 1.05, 1.18, 1.12], season=.08, temp=.006, promo=.8, holiday=.16, noise=.11),
    dict(id="chicken", name="Chicken Breast 1kg", category="Meat", legacy_model="Random Forest", base=74, price=8.90, cost=6.10,
         lead=2, shelf=5, stock_days=1.4, on_order=0, max_cover=4,
         weekly=[.82, .84, .9, .98, 1.12, 1.28, 1.3], season=.04, temp=0.0, promo=.55, holiday=.2, noise=.12),
    dict(id="apples", name="Gala Apples 1kg", category="Produce", legacy_model="XGBoost", base=105, price=3.40, cost=2.00,
         lead=2, shelf=18, stock_days=3.2, on_order=0, max_cover=12,
         weekly=[.92, .9, .94, .98, 1.03, 1.15, 1.14], season=.16, temp=.003, promo=.6, holiday=.12, noise=.10),
    dict(id="pasta", name="Pasta 500g", category="Staples", legacy_model="ARIMA", base=68, price=1.70, cost=0.95,
         lead=4, shelf=365, stock_days=12, on_order=0, max_cover=35,
         weekly=[.88, .9, .92, .96, 1.02, 1.16, 1.12], season=.04, temp=0.0, promo=.9, holiday=.18, noise=.09),
    dict(id="coffee", name="Ground Coffee 250g", category="Beverages", legacy_model="Prophet", base=46, price=6.80, cost=4.20,
         lead=5, shelf=270, stock_days=8, on_order=0, max_cover=30,
         weekly=[.94, .95, 1.0, 1.02, 1.06, 1.03, 1.0], season=.06, temp=0.0, promo=1.1, holiday=.3, noise=.12),
    dict(id="detergent", name="Laundry Detergent 2L", category="Household", legacy_model="XGBoost", base=31, price=8.50, cost=5.60,
         lead=7, shelf=730, stock_days=18, on_order=0, max_cover=60,
         weekly=[.9, .92, .95, .98, 1.0, 1.12, 1.1], season=.03, temp=0.0, promo=1.25, holiday=.35, noise=.10),
    dict(id="papertowels", name="Paper Towels 6 Roll Pack", category="Household", legacy_model="Random Forest", base=39, price=5.20, cost=3.10,
         lead=5, shelf=730, stock_days=9, on_order=0, max_cover=40,
         weekly=[.9, .91, .94, .98, 1.0, 1.16, 1.14], season=.02, temp=0.0, promo=.85, holiday=.25, noise=.09),
]


def _nth_weekday(year, month, weekday, n):
    d = pd.Timestamp(year=year, month=month, day=1)
    d += pd.Timedelta(days=(weekday - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return d - pd.Timedelta(days=(d.weekday() - weekday) % 7)


def holiday_table(years):
    """US calendar of holidays and retail events used for enrichment.

    Each row is (name, date). The demand-lift derived from this table feeds the
    models automatically for both synthetic and uploaded data.
    """
    rows = []
    for y in years:
        thanksgiving = _nth_weekday(y, 11, 3, 4)
        rows += [
            ("New Year", pd.Timestamp(y, 1, 1)),
            ("Martin Luther King Jr. Day", _nth_weekday(y, 1, 0, 3)),
            ("Valentine's Day", pd.Timestamp(y, 2, 14)),
            ("Presidents' Day", _nth_weekday(y, 2, 0, 3)),
            ("Memorial Day", _last_weekday(y, 5, 0)),
            ("Juneteenth", pd.Timestamp(y, 6, 19)),
            ("Independence Day", pd.Timestamp(y, 7, 4)),
            ("Labor Day", _nth_weekday(y, 9, 0, 1)),
            ("Halloween", pd.Timestamp(y, 10, 31)),
            ("Columbus Day", _nth_weekday(y, 10, 0, 2)),
            ("Veterans Day", pd.Timestamp(y, 11, 11)),
            ("Thanksgiving", thanksgiving),
            ("Black Friday", thanksgiving + pd.Timedelta(days=1)),
            ("Cyber Monday", thanksgiving + pd.Timedelta(days=4)),
            ("Christmas", pd.Timestamp(y, 12, 25)),
            ("New Year's Eve", pd.Timestamp(y, 12, 31)),
        ]
    return pd.DataFrame(rows, columns=["holiday", "ds"])


def _holiday_lift(dates, hol):
    """Ramp-up before the holiday (7 days), spike on the day, small dip after."""
    out = np.zeros(len(dates))
    for d in hol["ds"]:
        delta = (dates - d).days.values
        out += np.where((delta >= -7) & (delta <= 0), np.exp(delta / 3.0), 0)
        out -= np.where((delta > 0) & (delta <= 2), .15, 0)
    return out


def generate(seed=SEED):
    """Returns full timeline (history + HORIZON future days), long format."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end - pd.Timedelta(days=3 * 365 - 1), end + pd.Timedelta(days=HORIZON))
    n = len(dates)
    hol = holiday_table(range(dates[0].year, dates[-1].year + 1))
    hl = _holiday_lift(dates, hol)
    doy = dates.dayofyear.values
    # temperature (deg C): annual cycle + weather noise; forecast days use the same generator (weather forecast)
    temp = 22 + 9 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 2.2, n)
    frames = []
    for p in PRODUCTS:
        # planned promotions: blocks of 3-7 days, ~14% of days, discount 10-30%
        disc = np.zeros(n)
        i = int(rng.integers(5, 20))
        while i < n:
            L = int(rng.integers(3, 8)); disc[i:i + L] = rng.choice([.10, .15, .20, .25, .30])
            i += L + int(rng.integers(14, 40))
        season = 1 + p["season"] * np.sin(2 * np.pi * (doy - 100) / 365)
        trend = 1 + 0.05 * np.arange(n) / 365            # slow ~5%/yr growth
        weekly = np.array(p["weekly"])[dates.dayofweek.values]
        tempfx = np.exp(p["temp"] * (temp - 22))
        lam = (p["base"] * trend * weekly * season * tempfx * (1 + p["holiday"] * hl)
               * (1 + p["promo"] * disc * 3) * rng.lognormal(0, p["noise"], n))
        sales = rng.poisson(np.maximum(lam, 1)).astype(float)
        f = pd.DataFrame(dict(date=dates, product=p["id"], sales=sales, discount=disc,
                              price=np.round(p["price"] * (1 - disc), 2), temp=np.round(temp, 1),
                              holiday_lift=hl))
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df, end, hol
