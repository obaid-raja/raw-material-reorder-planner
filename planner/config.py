"""Planning assumptions for the simulated feed mill.

Every number here is an illustrative assumption for a mid-size poultry feed mill.
None of it is real data from any company.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    key: str                # column name used in the datasets
    name: str               # label for charts and tables
    inclusion: float        # share of finished feed by weight
    lead_time_weeks: int    # weeks from placing an order to delivery
    capacity_t: float       # storage available for this material (tonnes)
    min_order_t: float      # smallest order a supplier will deliver (tonnes)
    price_column: str       # column in the World Bank Pink Sheet


MATERIALS = [
    Material("maize", "Maize", 0.55, 3, 2500, 500, "Maize"),
    Material("soybean_meal", "Soybean meal", 0.25, 6, 1500, 250, "Soybean meal"),
    Material("wheat", "Wheat", 0.10, 4, 600, 100, "Wheat, US HRW"),
]
MATERIAL_BY_KEY = {m.key: m for m in MATERIALS}

# Simulation window: weekly, Mondays. The World Bank file used runs to Dec 2024.
START = "2021-01-04"
END = "2024-12-30"
TEST_WEEKS = 52              # the final 52 weeks (2024) are held out for testing
FORECAST_HORIZON = 12        # weeks ahead each forecast looks
SERVICE_Z = 1.65             # safety stock factor, roughly a 95% cycle service level
PLAN_WEEKS = 12              # length of the forward order plan

# Reactive buying: urgent purchases from local traders on the spot market.
SPOT_LEAD_TIME_WEEKS = 1     # faster than a planned contract order...
SPOT_PREMIUM = 0.08          # ...but 8% above the market price
REACTIVE_TRIGGER_WEEKS = 1.5 # buy when stock + open orders cover less than 1.5 weeks of use
REACTIVE_COVER_WEEKS = 3.0   # and buy 3 weeks of use
