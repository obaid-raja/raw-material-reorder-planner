import numpy as np

from planner import forecast as fc
from planner.inventory import simulate

WEEKS = 60
DEMAND = np.full(WEEKS, 100.0)
PRICES = np.full(WEEKS, 200.0)
PERFECT = np.vstack([np.full(12, 100.0) for _ in range(WEEKS)])


def run(policy, **kw):
    args = dict(demand=DEMAND, prices=PRICES, lead_time=3, capacity=1000, min_order=50,
                policy=policy, initial_stock=400, forecasts=PERFECT)
    args.update(kw)
    return simulate(**args)[0]


def test_perfect_forecast_never_runs_out():
    weekly = run("planned")
    assert weekly["short_t"].max() < 1e-9


def test_stock_never_negative():
    for policy in ("planned", "reactive"):
        assert (run(policy)["stock_end_t"] >= -1e-9).all()


def test_storage_limit_respected():
    weekly = run("planned", capacity=450, safety=300)    # buffer wants more than fits
    assert weekly["stock_after_delivery_t"].max() <= 450 + 1e-9


def test_orders_respect_minimum():
    orders = run("planned", min_order=250)["order_t"]
    assert ((orders == 0) | (orders >= 250)).all()


def test_lead_time_timing():
    """An order placed at the end of week t arrives at the start of week t + L + 1."""
    weekly = run("planned", initial_stock=0)
    first_order = int(np.argmax(weekly["order_t"].to_numpy() > 0))
    assert weekly["arrived_t"].iloc[first_order + 4] > 0
    assert weekly["arrived_t"].iloc[first_order + 1: first_order + 4].sum() == 0


def test_seasonal_naive_repeats_last_year():
    hist = np.arange(104, dtype=float)
    assert np.allclose(fc.seasonal_naive(hist, 3), hist[52:55])


def test_accuracy_zero_for_perfect_forecast():
    series = np.full(80, 50.0)
    F = np.full((20, 6), 50.0)
    acc = fc.accuracy(series, F, first_origin=59, lead_time=3)
    assert acc["weekly_mape_pct"] == 0 and acc["lead_time_mape_pct"] == 0
