"""Run the full analysis: data -> forecasts -> backtest -> policy comparison -> order plan -> charts.

Usage:
    python run_pipeline.py          rebuild everything (about 2 minutes)
    python run_pipeline.py --fast   reuse the saved rolling forecasts
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from planner import data, pipeline
from planner.config import MATERIALS

OUT = Path(__file__).parent / "outputs"
FIG, TAB = OUT / "figures", OUT / "tables"
BLUE, ORANGE, GREY, RED = "#1f5f8b", "#d9822b", "#8a8a8a", "#c0392b"
POLICY_LABELS = {"reactive": "Reactive", "rule_of_thumb": "Rule of thumb", "planned": "Forecast-planned"}
POLICY_COLOURS = {"reactive": ORANGE, "rule_of_thumb": GREY, "planned": BLUE}


def policy_args(name, best_model):
    """The three policies compared. Rule of thumb = planned logic, flat average, no buffer."""
    return {
        "reactive": dict(policy="reactive"),
        "rule_of_thumb": dict(policy="planned", model="moving_average", z=0.0),
        "planned": dict(policy="planned", model=best_model),
    }[name]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    consumption, prices = data.build() if data.RAW_PRICES.exists() else data.load()
    if "--fast" in sys.argv:
        forecasts, sigmas = pipeline.load_forecasts()
    else:
        print("Computing rolling forecasts (Holt-Winters refits weekly; takes a minute or two)...")
        forecasts, sigmas = pipeline.compute_forecasts(consumption)
        pipeline.save_forecasts(forecasts, sigmas)

    acc = pipeline.accuracy_table(consumption, forecasts)
    acc.round(2).to_csv(TAB / "forecast_accuracy.csv", index=False)
    best = pipeline.best_models(acc)

    runs, summary, plans = {}, [], []
    for policy in POLICY_LABELS:
        runs[policy] = {}
        for m in MATERIALS:
            weekly, state, s = pipeline.run_policy(
                consumption, prices, forecasts, sigmas, m.key, **policy_args(policy, best[m.key]))
            runs[policy][m.key] = (weekly, state)
            summary.append({"policy": POLICY_LABELS[policy], "material": m.name, **s})
            weekly.to_csv(TAB / f"weekly_{policy}_{m.key}.csv", index=False, float_format="%.1f")
    summary = pd.DataFrame(summary)
    summary.round(1).to_csv(TAB / "policy_comparison.csv", index=False)

    impact = {p: pipeline.production_impact(consumption["feed_t"], {k: w for k, (w, _) in runs[p].items()})
              for p in runs}
    prod = pd.DataFrame([{
        "policy": POLICY_LABELS[p], "weeks_with_lost_output": int((d["lost_feed_t"] > 0.01).sum()),
        "lost_feed_t": d["lost_feed_t"].sum(), "lost_feed_bags_50kg": d["lost_feed_t"].sum() * 20,
        "lost_pct_of_plan": d["lost_feed_t"].sum() / d["planned_feed_t"].sum() * 100,
    } for p, d in impact.items()])
    prod.round(1).to_csv(TAB / "production_impact.csv", index=False)

    for m in MATERIALS:
        plans.append(pipeline.forward_plan(consumption, prices, m.key, best[m.key], runs["planned"][m.key][1]))
    plan = pd.concat(plans, ignore_index=True)
    plan.to_csv(TAB / "order_plan_next_12_weeks.csv", index=False)

    charts(consumption, forecasts, acc, runs, impact, best)
    print("\nBest model per material:", best)
    print("\nForecast accuracy:\n", acc.round(1).to_string(index=False))
    print("\nPolicy comparison:\n", summary.round(1).to_string(index=False))
    print("\nProduction impact:\n", prod.round(1).to_string(index=False))
    print("\nOrder plan:\n", plan.to_string(index=False))


def charts(consumption, forecasts, acc, runs, impact, best):
    n_train = pipeline.split(consumption)
    test_start = consumption.index[n_train]

    # 1. Demand history
    fig, ax = plt.subplots(figsize=(10, 4))
    for m, c in zip(MATERIALS, (BLUE, ORANGE, GREY)):
        ax.plot(consumption.index, consumption[m.key], color=c, lw=1.4, label=m.name)
    ax.axvspan(test_start, consumption.index[-1], color="#eeeeee", zorder=0)
    ax.text(test_start, ax.get_ylim()[1] * 0.97, "  test year (held out)", va="top", color="#555")
    style(ax, "Weekly raw-material consumption (simulated mill)", "tonnes per week")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "1_consumption.png", dpi=150); plt.close(fig)

    # 2. Forecast vs actual, maize, lead-time-ahead forecasts
    m = MATERIALS[0]
    F = pipeline.forecast_matrix(forecasts, m.key, best[m.key])
    L = m.lead_time_weeks
    fc_dates = consumption.index[n_train - 1 + L: n_train - 1 + L + len(F)]
    fig, ax = plt.subplots(figsize=(10, 4))
    test = consumption[m.key].iloc[n_train:]
    ax.plot(test.index, test, color="black", lw=1.6, label="Actual")
    ax.plot(fc_dates[: len(F)], F[: len(fc_dates), L - 1], color=BLUE, lw=1.6, ls="--",
            label=f"Forecast made {L} weeks earlier")
    style(ax, f"{m.name}: forecast {L} weeks ahead vs actual, 2024", "tonnes per week")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "2_forecast_vs_actual_maize.png", dpi=150); plt.close(fig)

    # 3. Accuracy by model
    pivot = acc.pivot(index="model", columns="material", values="lead_time_mape_pct")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot.plot.barh(ax=ax, color=[BLUE, ORANGE, GREY][: len(pivot.columns)], width=0.75)
    ax.set_xlabel("error in lead-time demand (MAPE, %) - lower is better")
    ax.set_ylabel("")
    ax.set_title("Forecast error over each material's lead time, test year", loc="left",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "3_forecast_accuracy.png", dpi=150); plt.close(fig)

    # 4. Inventory trajectory, maize, both policies
    fig, ax = plt.subplots(figsize=(10, 4))
    for policy in ("reactive", "planned"):
        w = runs[policy][m.key][0]
        ax.plot(w["week"], w["stock_after_delivery_t"], color=POLICY_COLOURS[policy], lw=1.6,
                label=POLICY_LABELS[policy])
        out = w[w["short_t"] > 0.01]
        if len(out):
            ax.scatter(out["week"], [0] * len(out), color=POLICY_COLOURS[policy], s=22, zorder=5,
                       marker="v", label=f"{POLICY_LABELS[policy]}: stock-out week")
    ax.axhline(m.capacity_t, color="black", lw=1, ls=":")
    ax.text(w["week"].iloc[0], m.capacity_t, " storage limit", va="bottom")
    ax.set_ylim(-150, m.capacity_t * 1.35)
    style(ax, f"{m.name} stock at the start of each week (after deliveries), 2024", "tonnes")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "4_stock_maize.png", dpi=150); plt.close(fig)

    # 5. Lost production, total per policy
    fig, ax = plt.subplots(figsize=(10, 3.2))
    names = list(POLICY_LABELS)
    totals = [impact[p]["lost_feed_t"].sum() for p in names]
    pct = [impact[p]["lost_feed_t"].sum() / impact[p]["planned_feed_t"].sum() * 100 for p in names]
    bars = ax.barh([POLICY_LABELS[p] for p in names], totals, color=[POLICY_COLOURS[p] for p in names])
    for b, t, p in zip(bars, totals, pct):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {t:,.0f} t ({p:.1f}% of planned output)",
                va="center")
    ax.invert_yaxis()
    ax.set_xlim(0, max(totals) * 1.45)
    ax.set_xlabel("tonnes of feed not produced because an ingredient ran out")
    ax.set_title("Feed output lost to raw-material shortages, 2024", loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "5_lost_production.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
