"""Build the two datasets: simulated weekly consumption and real World Bank prices."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import END, MATERIALS, START

ROOT = Path(__file__).resolve().parents[1]
RAW_PRICES = ROOT / "data" / "raw" / "CMO-Historical-Data-Monthly.xlsx"
PROCESSED = ROOT / "data" / "processed"
PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/"
    "related/CMO-Historical-Data-Monthly.xlsx"
)


def weeks() -> pd.DatetimeIndex:
    return pd.date_range(START, END, freq="W-MON", name="week")


def simulate_production(seed: int = 42) -> pd.Series:
    """Weekly finished-feed output in tonnes: growth trend x yearly seasonality x noise."""
    idx = weeks()
    t = np.arange(len(idx))
    rng = np.random.default_rng(seed)
    trend = 650 * 1.06 ** (t / 52)                      # ~650 t/week, growing 6% a year
    season = 1 + 0.10 * np.sin(2 * np.pi * (t - 8) / 52) + 0.04 * np.sin(4 * np.pi * t / 52)
    noise = rng.normal(1, 0.06, len(idx))
    return pd.Series(trend * season * noise, index=idx, name="feed_t")


def simulate_consumption(seed: int = 7) -> pd.DataFrame:
    """Raw-material use = feed output x inclusion rate, with small formulation noise."""
    feed = simulate_production()
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"feed_t": feed})
    for m in MATERIALS:
        df[m.key] = feed * m.inclusion * rng.normal(1, 0.03, len(feed))
    return df.round(1)


def load_monthly_prices(path: Path = RAW_PRICES) -> pd.DataFrame:
    """Monthly prices in US$ per tonne from the World Bank Pink Sheet."""
    raw = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
    raw = raw.rename(columns={raw.columns[0]: "month"})
    raw = raw[raw["month"].astype(str).str.fullmatch(r"\d{4}M\d{2}")].copy()
    raw["month"] = pd.to_datetime(raw["month"].astype(str).str.replace("M", "-") + "-01")
    cols = {m.price_column: m.key for m in MATERIALS}
    return raw.set_index("month")[list(cols)].rename(columns=cols).apply(pd.to_numeric, errors="coerce")


def weekly_prices(monthly: pd.DataFrame) -> pd.DataFrame:
    """Each week takes the average price of its calendar month."""
    idx = weeks()
    out = monthly.reindex(idx.to_period("M").to_timestamp())
    out.index = idx
    return out.round(2)


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    consumption = simulate_consumption()
    prices = weekly_prices(load_monthly_prices())
    if prices.isna().any().any():
        raise ValueError("Missing prices in the simulation window; re-download the Pink Sheet.")
    consumption.to_csv(PROCESSED / "weekly_consumption.csv")
    prices.to_csv(PROCESSED / "weekly_prices.csv")
    return consumption, prices


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    read = lambda name: pd.read_csv(PROCESSED / name, index_col="week", parse_dates=["week"])
    return read("weekly_consumption.csv"), read("weekly_prices.csv")
