"""Weekly demand forecasts, rolling-origin backtests and accuracy metrics."""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON = 52


def naive(hist: np.ndarray, h: int) -> np.ndarray:
    """Next weeks = last week."""
    return np.repeat(hist[-1], h)


def moving_average(hist: np.ndarray, h: int, k: int = 4) -> np.ndarray:
    """Next weeks = average of the last k weeks."""
    return np.repeat(hist[-k:].mean(), h)


def seasonal_naive(hist: np.ndarray, h: int) -> np.ndarray:
    """Next weeks = the same weeks one year earlier."""
    return np.array([hist[-SEASON + (i % SEASON)] for i in range(h)])


def _fit_holt_winters(hist: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(
            hist, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=SEASON, initialization_method="estimated",
        ).fit()


def holt_winters(hist: np.ndarray, h: int) -> np.ndarray:
    """Exponential smoothing with a damped trend and yearly seasonality."""
    return np.asarray(_fit_holt_winters(hist).forecast(h))


MODELS = {
    "naive": naive,
    "moving_average": moving_average,
    "seasonal_naive": seasonal_naive,
    "holt_winters": holt_winters,
}
MODEL_LABELS = {
    "naive": "Naive (last week)",
    "moving_average": "4-week moving average",
    "seasonal_naive": "Seasonal naive (same week last year)",
    "holt_winters": "Holt-Winters (trend + seasonality)",
}


def one_step_sigma(train: np.ndarray, model: str) -> float:
    """Standard deviation of one-week-ahead errors on the training data.

    This sizes the safety stock: the less predictable the demand, the more buffer.
    """
    s = pd.Series(train)
    if model == "naive":
        err = s - s.shift(1)
    elif model == "moving_average":
        err = s - s.rolling(4).mean().shift(1)
    elif model == "seasonal_naive":
        err = s - s.shift(SEASON)
    elif model == "holt_winters":
        err = pd.Series(_fit_holt_winters(train).resid)
    else:
        raise KeyError(model)
    return float(err.dropna().std())


def rolling_forecasts(series: np.ndarray, model: str, first_origin: int, horizon: int) -> np.ndarray:
    """Row i holds the forecast made at the end of week (first_origin + i), using only data
    up to that week, for the following `horizon` weeks. Nothing from the future leaks in."""
    fn = MODELS[model]
    return np.vstack([fn(series[: o + 1], horizon) for o in range(first_origin, len(series))])


def accuracy(series: np.ndarray, forecasts: np.ndarray, first_origin: int, lead_time: int) -> dict:
    """Weekly accuracy across horizons 1..H, plus accuracy of total demand over the lead time,
    which is the number a buyer actually relies on."""
    errs, acts = [], []
    lt_err, lt_act = [], []
    n = len(series)
    for i, f in enumerate(forecasts):
        o = first_origin + i
        for h in range(1, len(f) + 1):
            if o + h < n:
                errs.append(f[h - 1] - series[o + h])
                acts.append(series[o + h])
        if o + lead_time < n:
            actual_total = series[o + 1: o + 1 + lead_time].sum()
            lt_err.append(f[:lead_time].sum() - actual_total)
            lt_act.append(actual_total)
    errs, acts = np.array(errs), np.array(acts)
    lt_err, lt_act = np.array(lt_err), np.array(lt_act)
    return {
        "weekly_mape_pct": float(np.mean(np.abs(errs) / acts) * 100),
        "weekly_mae_t": float(np.mean(np.abs(errs))),
        "bias_pct": float(errs.sum() / acts.sum() * 100),
        "lead_time_mape_pct": float(np.mean(np.abs(lt_err) / lt_act) * 100),
    }
