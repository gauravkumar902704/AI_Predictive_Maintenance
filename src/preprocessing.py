"""
preprocessing.py
----------------
Data load karna, clean karna aur feature engineering.

Sabse important baat: training aur Flask prediction DONO yahi functions
use karte hain, taaki features dono jagah bilkul same bane
(notebook vs app wali problem na aaye).

Leakage se bachne ke liye saare rolling/diff features sirf PICHLI
readings se bante hain (pandas rolling backward-looking hoti hai).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Physical limits (Kelvin / rpm / Nm / min). Inse bahar ki value
# invalid maani jaayegi. Rare values ko hum delete nahi karte, kyunki
# woh warning signal ho sakti hain; sirf impossible values hatate hain.
VALID_RANGES = {
    config.COL_AIR_TEMP: (250, 400),
    config.COL_TEMP: (250, 450),
    config.COL_SPEED: (0, 10000),
    config.COL_TORQUE: (0, 500),
    config.COL_WEAR: (0, 1000),
}


# ------------------------------------------------------------------ load
def load_raw(path=None):
    """CSV padhta hai aur timestamp ko datetime me badalta hai."""
    path = path or config.RAW_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset nahi mila: {path}\n"
            "Pehle 'python -m src.data_generator' chalao ya apna CSV wahan rakho."
        )
    df = pd.read_csv(path)
    df[config.COL_TIME] = pd.to_datetime(df[config.COL_TIME], errors="coerce")
    return df


# ------------------------------------------------------------------ clean
def clean(df):
    """
    Basic cleaning:
      1. duplicate rows hatao
      2. numeric columns ko numbers me convert karo (galat text -> NaN)
      3. impossible values ko NaN banao
      4. missing values ko machine-wise forward fill / median se bharo
      5. timestamp na ho to row hata do, phir machine + time ke hisaab se sort
    """
    df = df.copy()
    df = df.drop_duplicates()

    for col in VALID_RANGES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = VALID_RANGES[col]
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df = df.dropna(subset=[config.COL_TIME, config.COL_MACHINE])
    df = df.sort_values([config.COL_MACHINE, config.COL_TIME]).reset_index(drop=True)

    # sensor jaisa data hai, isliye pehle usi machine ki pichli value se fill karte hain
    numeric_cols = list(VALID_RANGES.keys())
    df[numeric_cols] = df.groupby(config.COL_MACHINE)[numeric_cols].ffill()
    # jo abhi bhi khali hain (machine ki pehli reading) unko overall median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df[config.COL_TYPE] = df[config.COL_TYPE].fillna("L")
    return df


# ------------------------------------------------------------------ features
def _rolling_slope(series, window):
    """Last 'window' values par straight line fit karke slope nikalta hai."""
    def slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return float(np.polyfit(x, values, 1)[0])
    return series.rolling(window, min_periods=2).apply(slope, raw=True)


def add_features(df):
    """
    Raw sensor values se engineered features banata hai.
    Input df me machine_id aur timestamp hona zaroori hai aur wo
    machine + time ke hisaab se sorted honi chahiye.
    """
    df = df.copy()
    g = df.groupby(config.COL_MACHINE)
    w = config.ROLLING_WINDOW

    # ---- time features
    df["hour"] = df[config.COL_TIME].dt.hour
    df["day_of_week"] = df[config.COL_TIME].dt.dayofweek

    # ---- machine type ko number banao
    df["type_code"] = df[config.COL_TYPE].map(TYPE_MAP).fillna(0).astype(int)

    # ---- simple relationship features
    df["temp_diff"] = df[config.COL_TEMP] - df[config.COL_AIR_TEMP]
    # power = torque * angular speed (watt)
    df["power"] = df[config.COL_TORQUE] * df[config.COL_SPEED] * 2 * np.pi / 60.0
    # 1e-6 isliye ki zero se divide na ho
    df["torque_speed_ratio"] = df[config.COL_TORQUE] / (df[config.COL_SPEED] + 1e-6)

    # ---- rolling / trend features (sirf past data)
    df["temp_roll_mean"] = g[config.COL_TEMP].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["temp_roll_std"] = g[config.COL_TEMP].transform(lambda s: s.rolling(w, min_periods=2).std())
    df["temp_delta"] = g[config.COL_TEMP].diff()
    df["temp_slope"] = g[config.COL_TEMP].transform(lambda s: _rolling_slope(s, config.SLOPE_WINDOW))

    df["torque_roll_mean"] = g[config.COL_TORQUE].transform(lambda s: s.rolling(w, min_periods=1).mean())
    df["torque_delta"] = g[config.COL_TORQUE].diff()
    df["speed_roll_std"] = g[config.COL_SPEED].transform(lambda s: s.rolling(w, min_periods=2).std())

    # wear ek dum 0 ho jaye (tool replace) to negative diff aayega, usko 0 maan lete hain
    df["wear_rate"] = g[config.COL_WEAR].diff().clip(lower=0)

    # shuru ki rows me diff/std NaN hote hain -> 0 se bhar do
    fill_zero = ["temp_roll_std", "temp_delta", "temp_slope",
                 "torque_delta", "speed_roll_std", "wear_rate"]
    df[fill_zero] = df[fill_zero].fillna(0.0)
    return df


def add_rul(df, cap=None):
    """
    Remaining Useful Life (RUL): agle failure me kitni readings bachi hain.
    Har machine ke liye peeche se count karte hain. Jis machine me aage
    koi failure nahi aaya uski RUL cap value maani jaati hai.
    """
    cap = cap or config.RUL_CAP
    df = df.copy()
    rul = np.full(len(df), cap, dtype=float)

    for _, idx in df.groupby(config.COL_MACHINE).indices.items():
        fails = df[config.COL_FAILURE].values[idx]
        next_fail = None
        # peeche se aage chalte hain
        for pos in range(len(idx) - 1, -1, -1):
            if fails[pos] == 1:
                next_fail = pos
            if next_fail is not None:
                rul[idx[pos]] = min(next_fail - pos, cap)
    df["rul"] = rul
    return df


def build_training_frame(path=None):
    """load -> clean -> features -> rul. Training ka full pipeline."""
    df = load_raw(path)
    df = clean(df)
    df = add_features(df)
    df = add_rul(df)
    return df


# ------------------------------------------------------------------ inference
def features_for_new_reading(history_df, reading):
    """
    Ek nayi reading ke liye features banata hai.

    history_df : us machine ki pichli readings (DataFrame, raw columns).
                 Khali bhi ho sakti hai (nayi machine).
    reading    : dict - nayi sensor reading.

    Pichli readings + nayi reading ko jodkar wahi add_features chalate hain
    jo training me chali thi, phir aakhri row return karte hain.
    """
    new_row = pd.DataFrame([reading])
    new_row[config.COL_TIME] = pd.to_datetime(new_row[config.COL_TIME])

    if history_df is not None and len(history_df) > 0:
        hist = history_df[new_row.columns.intersection(history_df.columns)].copy()
        hist[config.COL_TIME] = pd.to_datetime(hist[config.COL_TIME])
        combined = pd.concat([hist, new_row], ignore_index=True)
    else:
        combined = new_row

    combined = combined.sort_values([config.COL_MACHINE, config.COL_TIME]).reset_index(drop=True)
    combined = add_features(combined)
    return combined.iloc[[-1]]


# Error messages me dikhane ke liye field ke naam
FIELD_LABELS = {
    config.COL_TEMP: "Process temperature",
    config.COL_AIR_TEMP: "Air temperature",
    config.COL_SPEED: "Rotational speed",
    config.COL_TORQUE: "Torque",
    config.COL_WEAR: "Tool wear",
}


def validate_reading(values):
    """
    Form/API se aaya data check karta hai.
    Return: (cleaned_dict, errors_list)
    """
    errors = []
    cleaned = {}
    required = [config.COL_TEMP, config.COL_SPEED, config.COL_TORQUE, config.COL_WEAR]

    for key in required:
        raw = values.get(key)
        label = FIELD_LABELS[key]
        if raw is None or str(raw).strip() == "":
            errors.append(f"{label} is required.")
            continue
        try:
            cleaned[key] = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number (got {raw!r}).")

    # air temperature optional hai; na ho to process temp - 10 maan lete hain
    raw_air = values.get(config.COL_AIR_TEMP)
    if raw_air is None or str(raw_air).strip() == "":
        if config.COL_TEMP in cleaned:
            cleaned[config.COL_AIR_TEMP] = cleaned[config.COL_TEMP] - 10.0
    else:
        try:
            cleaned[config.COL_AIR_TEMP] = float(raw_air)
        except (TypeError, ValueError):
            errors.append(f"{FIELD_LABELS[config.COL_AIR_TEMP]} must be a number (got {raw_air!r}).")

    mtype = str(values.get(config.COL_TYPE, "L") or "L").upper()
    if mtype not in TYPE_MAP:
        errors.append("Machine type must be L, M or H.")
    cleaned[config.COL_TYPE] = mtype

    # physical range check
    for key, (lo, hi) in VALID_RANGES.items():
        if key in cleaned and not (lo <= cleaned[key] <= hi):
            errors.append(f"{FIELD_LABELS[key]} value {cleaned[key]:g} is outside the allowed range {lo}-{hi}.")

    return cleaned, errors
