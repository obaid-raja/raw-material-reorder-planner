"""Glue: forecasts for every material and model, policy runs, production impact, order plan."""
import numpy as np
import pandas as pd

from . import forecast as fc
from .config import (FORECAST_HORIZON, MATERIAL_BY_KEY, MATERIALS, PLAN_WEEKS, REACTIVE_COVER_WEEKS,
                     REACTIVE_TRIGGER_WEEKS, SERVICE_Z, SPOT_LEAD_TIME_WEEKS, SPOT_PREMIUM, TEST_WEEKS)
from .data import PROCESSED
from .inventory import State, simulate, summarise


def split(consumption: pd.DataFrame) -> int:
    """Index of the first test week."""
    return len(consumption) - TEST_WEEKS


def compute_forecasts(consumption: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling forecasts from the end of training to the last week, for every material x model."""
    n_train = split(consumption)
    origins = consumption.index[n_train - 1:]
    rows, sig = [], []
    for m in MATERIALS:
        series = consumption[m.key].to_numpy()
        for model in fc.MODELS:
            F = fc.rolling_forecasts(series, model, n_train - 1, FORECAST_HORIZON)
            for origin, f in zip(origins, F):
                for h, value in enumerate(f, start=1):
                    rows.append((m.key, model, origin, h, round(float(value), 2)))
            sig.append((m.key, model, fc.one_step_sigma(series[:n_train], model)))
    forecasts = pd.DataFrame(rows, columns=["material", "model", "origin_week", "h", "forecast_t"])
    sigmas = pd.DataFrame(sig, columns=["material", "model", "sigma_t"])
    return forecasts, sigmas


def save_forecasts(forecasts: pd.DataFrame, sigmas: pd.DataFrame) -> None:
    forecasts.to_csv(PROCESSED / "rolling_forecasts.csv", index=False)
    sigmas.to_csv(PROCESSED / "forecast_sigma.csv", index=False)


def load_forecasts() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_csv(PROCESSED / "rolling_forecasts.csv", parse_dates=["origin_week"])
    return f, pd.read_csv(PROCESSED / "forecast_sigma.csv")


def forecast_matrix(forecasts: pd.DataFrame, material: str, model: str) -> np.ndarray:
    sub = forecasts[(forecasts.material == material) & (forecasts.model == model)]
    return sub.pivot(index="origin_week", columns="h", values="forecast_t").sort_index().to_numpy()


def accuracy_table(consumption, forecasts) -> pd.DataFrame:
    n_train = split(consumption)
    out = []
    for m in MATERIALS:
        series = consumption[m.key].to_numpy()
        for model in fc.MODELS:
            F = forecast_matrix(forecasts, m.key, model)
            out.append({"material": m.name, "model": fc.MODEL_LABELS[model], "model_key": model,
                        **fc.accuracy(series, F, n_train - 1, m.lead_time_weeks)})
    return pd.DataFrame(out)


def best_models(acc: pd.DataFrame) -> dict:
    """Pick, per material, the model with the lowest lead-time MAPE."""
    by_name = {m.name: m.key for m in MATERIALS}
    best = acc.loc[acc.groupby("material")["lead_time_mape_pct"].idxmin()]
    return {by_name[r.material]: r.model_key for r in best.itertuples()}


def run_policy(consumption, prices, forecasts, sigmas, material, policy, model="holt_winters",
               lead_time=None, capacity=None, z=SERVICE_Z, min_order=None,
               spot_lead_time=SPOT_LEAD_TIME_WEEKS, spot_premium=SPOT_PREMIUM):
    """Simulate one material over the test year. Returns (weekly table, final state, summary).

    Planned orders use the material's contract lead time at market price; reactive orders
    go to the spot market (spot_lead_time, market price + spot_premium)."""
    m = MATERIAL_BY_KEY[material]
    if policy == "reactive":
        lead_time, premium = spot_lead_time, spot_premium
    else:
        lead_time, premium = (m.lead_time_weeks if lead_time is None else lead_time), 0.0
    capacity = m.capacity_t if capacity is None else capacity
    min_order = m.min_order_t if min_order is None else min_order
    n_train = split(consumption)
    series = consumption[material].to_numpy()
    demand = series[n_train:]
    price = prices[material].to_numpy()[n_train:]
    initial = (m.lead_time_weeks + 1) * series[n_train - 4: n_train].mean()
    F = forecast_matrix(forecasts, material, model)[1:]      # decision at end of test week t uses origin t
    sigma = float(sigmas.query("material == @material and model == @model")["sigma_t"].iloc[0])
    weekly, state = simulate(
        demand, price, lead_time, capacity, min_order, policy, initial,
        forecasts=F, safety=z * sigma * np.sqrt(lead_time + 1), premium=premium,
        trigger_weeks=REACTIVE_TRIGGER_WEEKS, cover_weeks=REACTIVE_COVER_WEEKS,
    )
    weekly.insert(0, "week", consumption.index[n_train:])
    return weekly, state, summarise(weekly, capacity)


def production_impact(feed_t: pd.Series, weekly_by_material: dict) -> pd.DataFrame:
    """A week's feed output is limited by the scarcest ingredient."""
    fill = pd.DataFrame({k: (w["used_t"] / w["demand_t"]).to_numpy() for k, w in weekly_by_material.items()})
    feed = feed_t.to_numpy()[-len(fill):]
    possible = fill.min(axis=1).to_numpy()
    return pd.DataFrame({"week": feed_t.index[-len(fill):], "planned_feed_t": feed,
                         "feed_fill": possible, "lost_feed_t": feed * (1 - possible)})


def forward_plan(consumption, prices, material, model, state: State, lead_time=None,
                 capacity=None, min_order=None, z=SERVICE_Z) -> pd.DataFrame:
    """Order plan for the next PLAN_WEEKS weeks after the data ends, continuing from the
    stock and open orders left at the end of the test year."""
    m = MATERIAL_BY_KEY[material]
    lead_time = m.lead_time_weeks if lead_time is None else lead_time
    capacity = m.capacity_t if capacity is None else capacity
    min_order = m.min_order_t if min_order is None else min_order
    series = consumption[material].to_numpy()
    path = fc.MODELS[model](series, PLAN_WEEKS + FORECAST_HORIZON + 1)
    carry = State(state.on_hand, {w - TEST_WEEKS: q for w, q in state.pipeline.items()})
    F = np.vstack([path[k + 1: k + 1 + FORECAST_HORIZON] for k in range(PLAN_WEEKS)])
    sigma_model = fc.one_step_sigma(series[: split(consumption)], model)
    last_price = float(prices[material].iloc[-1])
    weekly, _ = simulate(
        path[:PLAN_WEEKS], np.repeat(last_price, PLAN_WEEKS), lead_time, capacity,
        min_order, "planned", 0, forecasts=F,
        safety=z * sigma_model * np.sqrt(lead_time + 1), state=carry,
    )
    last = consumption.index[-1]
    weekly["week"] = [last + pd.Timedelta(weeks=k + 1) for k in range(PLAN_WEEKS)]
    orders = weekly[weekly["order_t"] > 0]
    return pd.DataFrame({
        "material": m.name,
        "order_by_week": orders["week"].dt.date,
        "quantity_t": orders["order_t"].round(0),
        "arrives_week": (orders["week"] + pd.Timedelta(weeks=lead_time + 1)).dt.date,
        "est_cost_usd": orders["order_cost_usd"].round(-2),
        "stock_end_of_week_t": orders["stock_end_t"].round(0),
    })
