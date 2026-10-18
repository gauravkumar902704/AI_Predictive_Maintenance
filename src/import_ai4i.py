"""
import_ai4i.py
--------------
Real AI4I 2020 Predictive Maintenance dataset (Kaggle / UCI) ko is project
ke format me badalta hai.

Kaise use karein:
    1. ai4i2020.csv download karo
    2. python -m src.import_ai4i path/to/ai4i2020.csv
    3. python -m src.train

Dhyan rakhne wali baat:
    AI4I me na machine_id hai na timestamp (har row alag independent sample hai).
    Isliye yahan rows ko order me hourly timestamp de diye jaate hain aur har
    'rows_per_machine' rows ko ek machine maan liya jaata hai. Rolling/trend
    features isliye is dataset par kam meaningful honge (ye limitation report
    me likhna chahiye).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# AI4I ke original column names -> hamare column names
RENAME = {
    "Type": config.COL_TYPE,
    "Air temperature [K]": config.COL_AIR_TEMP,
    "Process temperature [K]": config.COL_TEMP,
    "Rotational speed [rpm]": config.COL_SPEED,
    "Torque [Nm]": config.COL_TORQUE,
    "Tool wear [min]": config.COL_WEAR,
    "Machine failure": config.COL_FAILURE,
}

# Kaggle ke kuch versions me columns bina unit ke hote hain
# ("Air temperature" jaisa). Dono tarah ke headers chalenge.
ALT_NAMES = {
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
}

FAILURE_COLUMNS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


def convert(src_path, rows_per_machine=200):
    raw = pd.read_csv(src_path)
    raw = raw.rename(columns={k: v for k, v in ALT_NAMES.items() if k in raw.columns})
    missing = [c for c in RENAME if c not in raw.columns]
    if missing:
        raise ValueError(f"Ye columns file me nahi mile: {missing}. Kya ye AI4I 2020 ka CSV hai?")

    df = raw.rename(columns=RENAME)

    # failure type: jo bhi failure flag 1 hai uska naam, warna 'No Failure'
    ftype = pd.Series("No Failure", index=df.index)
    for col, name in FAILURE_COLUMNS.items():
        if col in df.columns:
            ftype[df[col] == 1] = name
    # kuch rows me 'Machine failure' = 1 hai par koi specific flag nahi (rare)
    ftype[(df[config.COL_FAILURE] == 1) & (ftype == "No Failure")] = "Random Failure"
    df[config.COL_FAILURE_TYPE] = ftype

    # machine id aur timestamp banao
    n = len(df)
    df[config.COL_MACHINE] = [f"M{(i // rows_per_machine) + 1:03d}" for i in range(n)]
    df[config.COL_TIME] = pd.date_range("2026-01-01", periods=n, freq="h")
    # time ko machine ke andar order me rakhne ke liye machine-wise 0..199 hours
    df[config.COL_TIME] = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        df.groupby(config.COL_MACHINE).cumcount(), unit="h")

    cols = [config.COL_MACHINE, config.COL_TIME, config.COL_TYPE, config.COL_AIR_TEMP,
            config.COL_TEMP, config.COL_SPEED, config.COL_TORQUE, config.COL_WEAR,
            config.COL_FAILURE, config.COL_FAILURE_TYPE]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="AI4I 2020 CSV ko project format me convert karo")
    parser.add_argument("csv", help="ai4i2020.csv ka path")
    parser.add_argument("--rows-per-machine", type=int, default=200)
    args = parser.parse_args()

    out = convert(args.csv, args.rows_per_machine)
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    out.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Saved {len(out)} rows -> {config.RAW_DATA_PATH}")
    print(f"Failure rate: {out[config.COL_FAILURE].mean() * 100:.2f}%")
    print("Ab 'python -m src.train' chalao. Purana data/maintenance.db delete kar dena "
          "taaki database naye data se dobara bhare.")


if __name__ == "__main__":
    main()
