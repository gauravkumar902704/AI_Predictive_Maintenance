"""
data_generator.py
-----------------
Industrial sensor data ka synthetic dataset banata hai (AI4I 2020 jaisa).

Kyun? Real dataset (AI4I 2020 / Kaggle) download karke seedha
data/raw/predictive_maintenance.csv me daal sakte ho. Jab tak woh nahi hai,
yeh script project ko chalane layak data bana deti hai.

Har machine ke liye time ke saath tool wear badhta hai, sensor values me
noise hota hai, aur kuch physical rules ke hisaab se failure aata hai:
    TWF - Tool Wear Failure
    HDF - Heat Dissipation Failure
    PWF - Power Failure
    OSF - Overstrain Failure
    RNF - Random Failure

Fleet me kuch machines "degrading" hoti hain: timeline ke end ki taraf unka
wear tezi se badhta hai, speed girti hai, torque badhta hai aur cooling kharab
hoti jaati hai. Aisi machines abhi tak fail nahi hui hoti, isliye model unhe
MEDIUM/HIGH risk dikhata hai, bilkul waise jaise asli plant me hota hai.

Run:
    python -m src.data_generator
"""
import os
import sys

import numpy as np
import pandas as pd

# project root ko path me daalte hain taaki "python src/data_generator.py" bhi chale
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Overstrain ki limit machine type ke hisaab se (tool_wear * torque)
OVERSTRAIN_LIMIT = {"L": 7500, "M": 8500, "H": 9500}

# Kitni machines degrade ho rahi hain (fraction)
DEGRADING_SHARE = 0.25


def generate_dataset(n_machines=50, readings_per_machine=200, seed=42):
    """Synthetic telemetry data banata hai aur DataFrame return karta hai."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01 00:00:00")
    rows = []

    for m in range(1, n_machines + 1):
        machine_id = f"M{m:03d}"
        # L zyada hoti hain, H kam (AI4I jaisa ratio)
        mtype = rng.choice(["L", "M", "H"], p=[0.6, 0.3, 0.1])

        wear = float(rng.integers(0, 60))         # purani/new machine
        base_air = rng.normal(300, 1.5)           # machine ka apna base temperature
        base_speed = rng.normal(1500, 60)

        # Degrading machine: ramp_start ke baad kharabi dheere dheere badhti hai
        degrading = rng.random() < DEGRADING_SHARE
        ramp_start = readings_per_machine - int(rng.integers(20, 45))
        ramp_len = readings_per_machine - ramp_start
        # timeline ke end tak kharabi kitni badh chuki hogi (1 = failure ke bahut paas)
        deg_end = rng.uniform(0.45, 0.9)

        for t in range(readings_per_machine):
            ts = start + pd.Timedelta(hours=t)

            # deg = 0 (theek) se 1 (bahut kharab) tak
            deg = 0.0
            if degrading and t >= ramp_start:
                deg = deg_end * (t - ramp_start + 1) / ramp_len

            # din-raat ka temperature cycle
            daily = 1.2 * np.sin(2 * np.pi * (ts.hour / 24.0))
            air = base_air + daily + rng.normal(0, 0.4)

            # wear zyada hone par process temperature thoda upar jaata hai,
            # lekin degrade hone par cooling kharab hoti hai (temp difference ghat-ta hai)
            drift = 0.004 * wear
            process = air + 9.3 + drift - 1.5 * deg + rng.normal(0, 0.6)

            speed = base_speed + rng.normal(0, 150) - 230 * deg
            speed = max(speed, 1100)

            # speed zyada -> torque kam (motor ki natural behaviour)
            torque = 40 - (speed - 1500) * 0.012 + 9 * deg + rng.normal(0, 4)
            torque = float(np.clip(torque, 3, 80))

            # ------- failure rules -------
            power = torque * speed * 2 * np.pi / 60.0   # watt
            failure_type = "No Failure"

            if wear > 190 and rng.random() < 0.30:
                failure_type = "Tool Wear Failure"
            elif (process - air) < 8.6 and speed < 1380:
                failure_type = "Heat Dissipation Failure"
            elif power < 3500 or power > 9000:
                failure_type = "Power Failure"
            elif wear * torque > OVERSTRAIN_LIMIT[mtype]:
                failure_type = "Overstrain Failure"
            elif rng.random() < 0.0015:
                failure_type = "Random Failure"

            failure = 0 if failure_type == "No Failure" else 1

            rows.append({
                config.COL_MACHINE: machine_id,
                config.COL_TIME: ts,
                config.COL_TYPE: mtype,
                config.COL_AIR_TEMP: round(air, 2),
                config.COL_TEMP: round(process, 2),
                config.COL_SPEED: int(round(speed)),
                config.COL_TORQUE: round(torque, 2),
                config.COL_WEAR: int(round(wear)),
                config.COL_FAILURE: failure,
                config.COL_FAILURE_TYPE: failure_type,
            })

            # agle step ke liye wear badhao; failure par tool replace ho jaata hai
            # (wear 0). Degrading machine ki kharabi tool badalne se theek nahi hoti
            # (cooling/speed ki problem baaki rehti hai), isliye degrade hoti rehti hai.
            # Pehle yahan degradation band ho jaati thi aur timeline ke end tak koi bhi
            # machine at-risk nahi bachti thi.
            if failure:
                wear = 0.0
            else:
                wear += rng.uniform(0.5, 3.5) + 2.5 * deg

    df = pd.DataFrame(rows)
    return df


def main():
    os.makedirs(os.path.dirname(config.RAW_DATA_PATH), exist_ok=True)
    df = generate_dataset()
    df.to_csv(config.RAW_DATA_PATH, index=False)
    rate = df[config.COL_FAILURE].mean() * 100
    print(f"Dataset saved: {config.RAW_DATA_PATH}")
    print(f"Rows: {len(df)} | Machines: {df[config.COL_MACHINE].nunique()} | Failure rate: {rate:.2f}%")
    print(df[config.COL_FAILURE_TYPE].value_counts())


if __name__ == "__main__":
    main()
