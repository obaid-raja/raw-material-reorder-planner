"""Week-by-week stock simulation for two buying policies.

Timing convention: an order placed at the end of week t arrives at the start of week
t + lead_time + 1, so it takes exactly `lead_time` full weeks.

- reactive: when stock on hand plus stock on order falls below `trigger_weeks` of recent use,
  buy `cover_weeks` of recent use urgently, typically from local traders on the spot market
  (short lead time, price premium). No forecast: "buy when it starts to look low".
- planned: every week, compare stock on hand plus stock on order against forecast demand
  until the next order could arrive (lead time + 1 week) plus safety stock. Order the gap,
  rounded up to the supplier's minimum, but never more than storage can hold on arrival.
  Run with a flat moving-average forecast and no safety stock, the same logic becomes the
  "rule of thumb" policy, which isolates what the forecast and the buffer add.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class State:
    on_hand: float
    pipeline: dict = field(default_factory=dict)   # arrival week index -> tonnes


def planned_order(t, state, forecast, lead_time, capacity, safety, min_order):
    on_order = sum(state.pipeline.values())
    cover = forecast[: lead_time + 1].sum()
    gap = cover + safety - (state.on_hand + on_order)
    if gap <= 0:
        return 0.0
    arrival = t + lead_time + 1
    arriving_first = sum(q for w, q in state.pipeline.items() if w <= arrival)
    expected_on_arrival = state.on_hand + arriving_first - forecast[:lead_time].sum()
    room = capacity - max(expected_on_arrival, 0.0)
    qty = min(max(gap, min_order), room)
    return float(qty) if qty >= min_order else 0.0


def reactive_order(t, state, recent_use, capacity, min_order, trigger_weeks, cover_weeks):
    position = state.on_hand + sum(state.pipeline.values())
    if position >= trigger_weeks * recent_use:
        return 0.0
    qty = min(max(cover_weeks * recent_use, min_order), capacity - state.on_hand)
    return float(qty) if qty >= min_order else 0.0


def simulate(
    demand: np.ndarray,
    prices: np.ndarray,
    lead_time: int,
    capacity: float,
    min_order: float,
    policy: str,
    initial_stock: float,
    forecasts: np.ndarray | None = None,
    safety: float = 0.0,
    trigger_weeks: float = 2.0,
    cover_weeks: float = 4.0,
    premium: float = 0.0,
    state: State | None = None,
) -> tuple[pd.DataFrame, State]:
    state = state or State(on_hand=float(initial_stock))
    rows = []
    for t in range(len(demand)):
        arrived = state.pipeline.pop(t, 0.0)
        state.on_hand += arrived
        start_stock = state.on_hand
        used = min(state.on_hand, demand[t])
        state.on_hand -= used
        if policy == "planned":
            qty = planned_order(t, state, forecasts[t], lead_time, capacity, safety, min_order)
        elif policy == "reactive":
            recent_use = float(np.mean(demand[max(0, t - 3): t + 1]))
            qty = reactive_order(t, state, recent_use, capacity, min_order, trigger_weeks, cover_weeks)
        else:
            raise KeyError(policy)
        if qty > 0:
            arrival = t + lead_time + 1
            state.pipeline[arrival] = state.pipeline.get(arrival, 0.0) + qty
        rows.append({
            "demand_t": demand[t], "used_t": used, "short_t": demand[t] - used,
            "arrived_t": arrived, "stock_after_delivery_t": start_stock,
            "stock_end_t": state.on_hand, "order_t": qty,
            "market_price_usd_t": prices[t], "order_cost_usd": qty * prices[t] * (1 + premium),
            "stock_value_usd": state.on_hand * prices[t],
        })
    return pd.DataFrame(rows), state


def summarise(weekly: pd.DataFrame, capacity: float) -> dict:
    return {
        "stockout_weeks": int((weekly["short_t"] > 0.01).sum()),
        "fill_rate_pct": float(weekly["used_t"].sum() / weekly["demand_t"].sum() * 100),
        "shortfall_t": float(weekly["short_t"].sum()),
        "orders": int((weekly["order_t"] > 0).sum()),
        "purchased_t": float(weekly["order_t"].sum()),
        "spend_usd": float(weekly["order_cost_usd"].sum()),
        "avg_price_paid_usd_t": float(weekly["order_cost_usd"].sum() / max(weekly["order_t"].sum(), 1e-9)),
        "avg_stock_t": float(weekly["stock_end_t"].mean()),
        "avg_stock_value_usd": float(weekly["stock_value_usd"].mean()),
        "peak_storage_use_pct": float(weekly["stock_after_delivery_t"].max() / capacity * 100),
    }
