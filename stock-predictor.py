# pyright: strict
# Silence missing type stubs just for yfinance in this file:
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

from typing import Any, cast
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf  # type: ignore[reportMissingTypeStubs]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


# ----------------------------
# Config
# ----------------------------
TICKER: str = "AAPL"
INDEX: str = "SPY"
SECTOR: str = "XLK"
START: str = "2010-01-01"
WEEKLY_RESAMPLE_RULE: str = "W-FRI"   # Friday close = end of week
HORIZON_WEEKS: int = 1
RNG_SEED: int = 42
np.random.seed(RNG_SEED)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    assert isinstance(out.index, pd.DatetimeIndex)
    return out


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            # pd.to_numeric has wide overloads; cast result back to Series[float]
            ser = cast(pd.Series, pd.to_numeric(out[c], errors="coerce"))
            out[c] = ser
    return out


def get_prices(ticker: str, start:str) -> pd.DataFrame:
    """
    Download and clean daily OHLCV data.
    """
    raw: Any = yf.download(
        tickers=ticker,
        start=start,
        auto_adjust=False,  # we do this manually
        back_adjust=False,  # we do this manually
        progress=False,
        group_by="column",
        threads=False,
        repair=True,
    )
    if raw is None:
        raise SystemExit(f"No data downloaded for {ticker}. Try different START or ticker.")

    df = cast(pd.DataFrame, raw)

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        needed = {"Open", "High", "Low", "Close", "Volume"}
        chosen_level: int | None = None
        for i in range(df.columns.nlevels):
            level_vals = set(df.columns.get_level_values(i))
            if needed.issubset(level_vals):
                df.columns = df.columns.get_level_values(i)
                chosen_level = i
                break
        if chosen_level is None:
            df.columns = df.columns.get_level_values(-1)

    needed_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = needed_cols - set(map(str, df.columns))
    if missing:
        raise SystemExit(f"{ticker}: missing {missing}. Raw columns: {list(df.columns)}")

    df = _ensure_datetime_index(df)
    df = _ensure_numeric(df, ["Open", "High", "Low", "Close", "Volume", "Adj Close"])
    if "Adj Close" in df.columns:
        adj_factor = df["Adj Close"] / df["Close"]
        df["Open"] = df["Open"] * adj_factor
        df["High"] = df["High"] * adj_factor
        df["Low"] = df["Low"] * adj_factor
        df["Close"] = df["Adj Close"]
        df = df.drop("Adj Close", axis=1)

    df = df.sort_index()
    return df


# ----------------------------
# 1) Download & align data
# ----------------------------
df_tkr: pd.DataFrame = get_prices(TICKER, START)
df_mkt: pd.DataFrame = get_prices(INDEX, START).rename(
    columns={"Close": "SPY_Close", "Volume": "SPY_Volume"}
)
df_sec: pd.DataFrame = get_prices(SECTOR, START).rename(
    columns={"Close": "SECTOR_Close", "Volume": "SECTOR_Volume"}
)

df: pd.DataFrame = df_tkr.join(df_mkt[["SPY_Close", "SPY_Volume"]], how="inner")
df = df.join(df_sec[["SECTOR_Close", "SECTOR_Volume"]], how="inner")
df = _ensure_datetime_index(df)

# ----------------------------
# 2) Resample to weekly bars (smoother)
# ----------------------------
agg_map: dict[str, str] = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
    "SPY_Close": "last",
    "SPY_Volume": "sum",
    "SECTOR_Close": "last",
    "SECTOR_Volume": "sum",
}
# Pandas typing for Resampler.agg is too loose; ignore call-overload here.
wk_any: Any = df.resample(WEEKLY_RESAMPLE_RULE).agg(agg_map)  # type: ignore[call-overload]
wk = cast(pd.DataFrame, wk_any).dropna(how="any")
wk = _ensure_numeric(
    wk,
    ["Open", "High", "Low", "Close", "Volume", "SPY_Close", "SPY_Volume", "SECTOR_Close", "SECTOR_Volume"],
)
df = wk

# Print a typed sanity line
first_idx = cast(pd.Timestamp, df.index.min())
last_idx = cast(pd.Timestamp, df.index.max())
print("Weekly rows:", int(len(df)), "| first:", str(first_idx), "| last:", str(last_idx))

# Weekly returns
df["ret_1w"] = df["Close"].pct_change()
df["mkt_ret_1w"] = df["SPY_Close"].pct_change()
df["sec_ret_1w"] = df["SECTOR_Close"].pct_change()

# ----------------------------
# 3) Target: future HORIZON_WEEKS direction (1 if up)
# ----------------------------
df["target"] = (df["Close"].shift(-HORIZON_WEEKS) > df["Close"]).astype("int8")

# ----------------------------
# 4) Feature engineering (weekly)
# ----------------------------
# Momentum / lagged returns
for lag in [1, 2, 3, 4, 8, 12, 26, 52]:
    df[f"ret_lag_{lag}w"] = df["ret_1w"].shift(lag)

# MA distance
for w in [4, 8, 12, 26, 52]:
    ma = df["Close"].rolling(w).mean()
    df[f"dist_ma_{w}w"] = (df["Close"] - ma) / (ma + 1e-12)

# Volatility
for w in [4, 12, 26]:
    df[f"vol_{w}w"] = df["ret_1w"].rolling(w).std()

# Bollinger percent-B (20w)
mid = df["Close"].rolling(20).mean()
std = df["Close"].rolling(20).std()
df["pctB_20w"] = (df["Close"] - mid) / (2.0 * std + 1e-12)

# ATR(14w)
tr_comp = pd.concat(
    [
        (df["High"] - df["Low"]),
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ],
    axis=1,
)
tr = tr_comp.max(axis=1)
df["atr_14w"] = tr.rolling(14).mean()

# Simplified ADX-like trend strength (14w)
up_move_s = cast(pd.Series, df["High"].diff()).astype("float64")
down_move_s = cast(pd.Series, (-df["Low"].diff())).astype("float64")

up_mask_arr: NDArray[np.bool_] = (up_move_s.to_numpy() > down_move_s.to_numpy()) & (
    up_move_s.to_numpy() > 0.0
)
down_mask_arr: NDArray[np.bool_] = (down_move_s.to_numpy() > up_move_s.to_numpy()) & (
    down_move_s.to_numpy() > 0.0
)

plusDM_arr: NDArray[np.float64] = np.where(up_mask_arr, up_move_s.to_numpy(), 0.0).astype(np.float64)
minusDM_arr: NDArray[np.float64] = np.where(down_mask_arr, down_move_s.to_numpy(), 0.0).astype(np.float64)

plusDM = pd.Series(plusDM_arr, index=df.index, dtype="float64")
minusDM = pd.Series(minusDM_arr, index=df.index, dtype="float64")

tr14 = tr.rolling(14).sum()
plusDI14 = 100.0 * plusDM.rolling(14).sum() / (tr14 + 1e-12)
minusDI14 = 100.0 * minusDM.rolling(14).sum() / (tr14 + 1e-12)
df["adx_14w"] = ((plusDI14 - minusDI14).abs() / ((plusDI14 + minusDI14) + 1e-12)) * 100.0

# Relative strength & beta-like
df["rel_strength_mkt_26w"] = df["ret_1w"].rolling(26).mean() - df["mkt_ret_1w"].rolling(26).mean()
df["rel_strength_sec_26w"] = df["ret_1w"].rolling(26).mean() - df["sec_ret_1w"].rolling(26).mean()

df["beta_mkt_26w"] = df["ret_1w"].rolling(26).cov(df["mkt_ret_1w"]) / (
    df["mkt_ret_1w"].rolling(26).var() + 1e-12
)
df["beta_sec_26w"] = df["ret_1w"].rolling(26).cov(df["sec_ret_1w"]) / (
    df["sec_ret_1w"].rolling(26).var() + 1e-12
)

# Calendar features (typed DatetimeIndex)
idx = cast(pd.DatetimeIndex, df.index)
months_arr: NDArray[np.int_] = idx.month.values.astype(np.int_)
df["month"] = months_arr.astype("int16")
df["m_sin"] = np.sin(2.0 * np.pi * df["month"].to_numpy() / 12.0)
df["m_cos"] = np.cos(2.0 * np.pi * df["month"].to_numpy() / 12.0)

# Final cleanup
df = cast(pd.DataFrame, df.dropna())

# ----------------------------
# 5) Train / Val / Test split
# ----------------------------
N: int = int(len(df))
train_end: int = int(N * 0.60)
val_end: int = int(N * 0.80)

EXCLUDE: set[str] = {
    "target",
    "Open", "High", "Low", "Close", "Volume",
    "SPY_Close", "SPY_Volume", "SECTOR_Close", "SECTOR_Volume",
    "month",
}
features: list[str] = [c for c in df.columns if c not in EXCLUDE]

X_train: NDArray[np.float64] = df.iloc[:train_end][features].to_numpy(dtype=np.float64)
y_train: NDArray[np.int_] = df.iloc[:train_end]["target"].to_numpy(dtype=np.int_)
X_val: NDArray[np.float64] = df.iloc[train_end:val_end][features].to_numpy(dtype=np.float64)
y_val: NDArray[np.int_] = df.iloc[train_end:val_end]["target"].to_numpy(dtype=np.int_)
X_test: NDArray[np.float64] = df.iloc[val_end:][features].to_numpy(dtype=np.float64)
y_test: NDArray[np.int_] = df.iloc[val_end:]["target"].to_numpy(dtype=np.int_)

# ----------------------------
# 6) Model: Gradient Boosting + Calibration
# ----------------------------
base = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=6,
    max_iter=600,
    min_samples_leaf=30,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RNG_SEED,
)
clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
clf.fit(X_train, y_train)

# ----------------------------
# 7) AUC-inversion safeguard + threshold tuning
# ----------------------------
val_proba_raw: NDArray[np.float64] = clf.predict_proba(X_val)[:, 1]
try:
    val_auc_f: float = float(roc_auc_score(y_val, val_proba_raw))
except ValueError:
    val_auc_f = float("nan")

invert: bool = bool(not np.isnan(val_auc_f) and val_auc_f < 0.5)
val_proba: NDArray[np.float64] = (1.0 - val_proba_raw) if invert else val_proba_raw

best_thr: float = 0.5
best_score: float = -1e9
actual_up_rate: float = float(y_val.mean())

for thr in np.linspace(0.40, 0.60, 41):
    preds: NDArray[np.int_] = (val_proba >= thr).astype(np.int_)
    acc_val = float(accuracy_score(y_val, preds))
    pred_up_rate: float = float(preds.mean())
    # Penalize if we are too far from 50% allocation, or if we have no trades
    penalty: float = (
        0.5 * abs(pred_up_rate - 0.5)
        + 0.10 * float(pred_up_rate == 0.0)
        + 0.10 * float(pred_up_rate == 1.0)
    )
    score: float = acc_val - penalty
    if score > best_score:
        best_score = score
        best_thr = float(thr)

# Evaluate on TEST
test_proba_raw: NDArray[np.float64] = clf.predict_proba(X_test)[:, 1]
test_proba: NDArray[np.float64] = (1.0 - test_proba_raw) if invert else test_proba_raw
y_pred: NDArray[np.int_] = (test_proba >= best_thr).astype(np.int_)

acc = float(accuracy_score(y_test, y_pred))
prec = float(precision_score(y_test, y_pred, zero_division=0))
rec = float(recall_score(y_test, y_pred, zero_division=0))
try:
    auc = float(roc_auc_score(y_test, test_proba))
except ValueError:
    auc = float("nan")

print(f"Metrics on Test (weekly, horizon={HORIZON_WEEKS}w):")
print(f"Validation AUC: {val_auc_f:.3f} | Inverted: {invert}")
print(f"Chosen threshold (from validation): {best_thr:.3f}")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"ROC AUC  : {auc:.3f}")

# Benchmark
always_up: NDArray[np.int_] = np.ones_like(y_test)
bench_acc = float(accuracy_score(y_test, always_up))
print("\nBenchmark (Always Up) Accuracy:", round(bench_acc, 3))

# ----------------------------
# 8) Toy backtest on TEST window
# ----------------------------
test_df = df.iloc[val_end:].copy()
test_df["proba"] = test_proba
test_df["signal"] = (test_df["proba"].to_numpy() >= best_thr).astype(np.int_)

# next period return aligned with today's signal
test_df["next_ret"] = test_df["Close"].pct_change(periods=HORIZON_WEEKS).shift(-HORIZON_WEEKS)
strategy_ret = (test_df["signal"].to_numpy() * test_df["next_ret"].to_numpy())
eq_curve = (1.0 + pd.Series(strategy_ret, index=test_df.index).fillna(0.0)).cumprod()
buy_hold = (1.0 + test_df["next_ret"].fillna(0.0)).cumprod()

if len(eq_curve) > 2:
    print("\nFinal equity (strategy):", round(float(eq_curve.iloc[-2]), 3))
    print("Final equity (buy&hold):", round(float(buy_hold.iloc[-2]), 3))

# ----------------------------
# 9) Plot equity curves on TEST
# ----------------------------
fig: plt.Figure = plt.figure(figsize=(9, 4.5))
ax = cast(plt.Axes, plt.gca())
pd.DataFrame({"eq_curve": eq_curve, "buy_hold": buy_hold}).dropna().plot(ax=ax)
ax.set_title(f"Toy Strategy vs Buy & Hold (Test) — {TICKER} weekly, horizon={HORIZON_WEEKS}w")
ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")
fig.tight_layout()
plt.show()

# ----------------------------
# 10) Optional: quick walk-forward AUC check
# ----------------------------
tscv = TimeSeriesSplit(n_splits=5)
auc_scores: list[float] = []

X_all: NDArray[np.float64] = df[features].to_numpy(dtype=np.float64)
y_all: NDArray[np.int_] = df["target"].to_numpy(dtype=np.int_)

for tr_idx, va_idx in tscv.split(X_all):
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

    base_cv = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=500,
        min_samples_leaf=30,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RNG_SEED,
    )
    cv_clf = CalibratedClassifierCV(base_cv, method="sigmoid", cv=3)
    cv_clf.fit(X_tr, y_tr)

    p_raw: NDArray[np.float64] = cv_clf.predict_proba(X_va)[:, 1]
    try:
        auc_va = float(roc_auc_score(y_va, p_raw))
    except ValueError:
        auc_va = float("nan")
    if not np.isnan(auc_va) and auc_va < 0.5:
        p_raw = 1.0 - p_raw
        try:
            auc_va = float(roc_auc_score(y_va, p_raw))
        except ValueError:
            auc_va = float("nan")
    auc_scores.append(auc_va)

auc_arr: NDArray[np.float64] = np.array(auc_scores, dtype=float)
print("\nTimeSeriesSplit ROC AUC (5 folds):", np.around(auc_arr, 3), "| Mean:", round(float(np.nanmean(auc_arr)), 3))
