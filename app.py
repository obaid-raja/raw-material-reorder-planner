"""Interactive reorder planner. Run locally with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from planner import data, pipeline
from planner import forecast as fc
from planner.config import (MATERIAL_BY_KEY, MATERIALS, SPOT_LEAD_TIME_WEEKS, SPOT_PREMIUM)

st.set_page_config(page_title="Raw-Material Reorder Planner", page_icon="📦", layout="wide")
COLOURS = {"Reactive": "#d9822b", "Rule of thumb": "#8a8a8a", "Forecast-planned": "#1f5f8b"}
SERVICE = {"90%": 1.28, "95%": 1.65, "99%": 2.33}


@st.cache_data
def load_all():
    consumption, prices = data.load()
    forecasts, sigmas = pipeline.load_forecasts()
    return consumption, prices, forecasts, sigmas


consumption, prices, forecasts, sigmas = load_all()

st.title("📦 Raw-Material Reorder Planner")
st.markdown(
    "When should a feed mill order maize, soybean meal and wheat, and how much, given supplier "
    "lead times and limited storage? Change the assumptions on the left and see how three buying "
    "policies would have performed over **2024**, a year the forecasts never saw.  \n"
    "*Consumption is simulated for an illustrative mid-size mill; prices are real World Bank "
    "monthly prices (US$/tonne).*"
)

with st.sidebar:
    st.header("Assumptions")
    key = st.selectbox("Raw material", [m.key for m in MATERIALS],
                       format_func=lambda k: MATERIAL_BY_KEY[k].name)
    m = MATERIAL_BY_KEY[key]
    st.subheader("Planned (contract) buying")
    lead = st.slider("Supplier lead time (weeks)", 1, 10, m.lead_time_weeks)
    cap = st.slider("Storage capacity (tonnes)", 200, 6000, int(m.capacity_t), step=100)
    moq = st.slider("Minimum order (tonnes)", 0, 1000, int(m.min_order_t), step=50)
    service = st.select_slider("Target service level", list(SERVICE), value="95%",
                               help="Sets the safety stock: higher = fewer stock-outs, more stock held.")
    model = st.selectbox("Forecast model", list(fc.MODELS), index=list(fc.MODELS).index("holt_winters"),
                         format_func=fc.MODEL_LABELS.get)
    st.subheader("Reactive (spot) buying")
    spot_lead = st.slider("Spot lead time (weeks)", 0, 4, SPOT_LEAD_TIME_WEEKS)
    premium = st.slider("Spot price premium (%)", 0, 25, int(SPOT_PREMIUM * 100)) / 100

common = dict(lead_time=lead, capacity=cap, min_order=moq)
runs = {
    "Reactive": pipeline.run_policy(consumption, prices, forecasts, sigmas, key, "reactive",
                                    capacity=cap, min_order=moq, spot_lead_time=spot_lead,
                                    spot_premium=premium),
    "Rule of thumb": pipeline.run_policy(consumption, prices, forecasts, sigmas, key, "planned",
                                         model="moving_average", z=0.0, **common),
    "Forecast-planned": pipeline.run_policy(consumption, prices, forecasts, sigmas, key, "planned",
                                            model=model, z=SERVICE[service], **common),
}

st.subheader(f"{m.name}: how each policy performed in 2024")
cols = st.columns(3)
for col, (name, (weekly, _, s)) in zip(cols, runs.items()):
    with col:
        st.markdown(f"**<span style='color:{COLOURS[name]}'>{name}</span>**", unsafe_allow_html=True)
        st.metric("Demand met from stock", f"{s['fill_rate_pct']:.1f}%")
        st.metric("Weeks with a stock-out", s["stockout_weeks"])
        st.metric("Average stock held", f"{s['avg_stock_t']:,.0f} t")
        st.metric("Average price paid", f"${s['avg_price_paid_usd_t']:,.0f}/t")

fig = go.Figure()
for name, (weekly, _, _) in runs.items():
    fig.add_scatter(x=weekly["week"], y=weekly["stock_after_delivery_t"], name=name,
                    line=dict(color=COLOURS[name], width=2))
fig.add_hline(y=cap, line_dash="dot", annotation_text="storage limit")
fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="tonnes in stock (start of week)",
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Forecast vs actual")
    n_train = pipeline.split(consumption)
    F = pipeline.forecast_matrix(forecasts, key, model)
    h = min(lead, F.shape[1])
    dates = consumption.index[n_train - 1 + h: n_train - 1 + h + len(F)]
    ff = go.Figure()
    test = consumption[key].iloc[n_train:]
    ff.add_scatter(x=test.index, y=test, name="Actual", line=dict(color="black"))
    ff.add_scatter(x=dates, y=F[: len(dates), h - 1], name=f"Forecast made {h} weeks earlier",
                   line=dict(color="#1f5f8b", dash="dash"))
    ff.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="tonnes per week",
                     legend=dict(orientation="h", y=1.12))
    st.plotly_chart(ff, width="stretch")
with right:
    st.subheader("Forecast accuracy, 2024")
    acc = pipeline.accuracy_table(consumption, forecasts)
    acc = acc[acc.material == m.name][["model", "weekly_mape_pct", "lead_time_mape_pct", "bias_pct"]]
    st.dataframe(
        acc.rename(columns={"model": "Model", "weekly_mape_pct": "Weekly error (MAPE %)",
                            "lead_time_mape_pct": "Lead-time demand error (MAPE %)", "bias_pct": "Bias %"})
        .round(1), hide_index=True, width="stretch")
    st.caption("Lead-time demand error is what matters for buying: how wrong was the forecast of total "
               f"use over the {m.lead_time_weeks}-week lead time? Bias below zero = under-forecasting.")

st.subheader("Order plan for the next 12 weeks (Jan to Mar 2025)")
plan = pipeline.forward_plan(consumption, prices, key, model, runs["Forecast-planned"][1], **common,
                             z=SERVICE[service])
if plan.empty:
    st.info("No orders needed in the next 12 weeks with these settings.")
else:
    st.dataframe(plan.drop(columns="material").rename(columns={
        "order_by_week": "Order by (week of)", "quantity_t": "Quantity (t)", "arrives_week": "Arrives (week of)",
        "est_cost_usd": "Est. cost (US$)", "stock_end_of_week_t": "Stock at end of order week (t)"}),
        hide_index=True, width="stretch")
    st.caption("Built from the forecast-planned policy, starting from the stock and open orders left at the "
               "end of 2024, priced at the latest World Bank monthly price.")

with st.expander("How the policies work"):
    st.markdown(f"""
- **Reactive:** when stock plus open orders falls below 1.5 weeks of recent use, buy 3 weeks' worth
  on the spot market ({spot_lead}-week delivery, {premium:.0%} premium). No forecast.
- **Rule of thumb:** each week, order enough to cover the next lead time + 1 week at the recent
  4-week average. No safety stock.
- **Forecast-planned:** same check, but uses the chosen forecast model plus safety stock sized from
  past forecast errors (service level {service}), and never orders more than storage can hold when
  the delivery arrives.
""")
