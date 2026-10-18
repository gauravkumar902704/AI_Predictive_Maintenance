"""
config.py
---------
Saari settings (paths, thresholds, column names) ek hi jagah rakhi hain
taaki baaki files me hardcode na karna pade.
"""
import os
import secrets

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "predictive_maintenance.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "predictive_maintenance.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
DB_PATH = os.path.join(BASE_DIR, "data", "maintenance.db")

# ---------------------------------------------------------------- raw columns
# Dataset (AI4I style) ke columns. Agar aap koi dusra CSV use karo to
# yahan naam badal dena.
COL_MACHINE = "machine_id"
COL_TIME = "timestamp"
COL_TYPE = "machine_type"          # L / M / H
COL_AIR_TEMP = "air_temperature"   # Kelvin
COL_TEMP = "temperature"           # process temperature, Kelvin
COL_SPEED = "rotational_speed"     # rpm
COL_TORQUE = "torque"              # Nm
COL_WEAR = "tool_wear"             # minutes
COL_FAILURE = "failure"            # 0 / 1
COL_FAILURE_TYPE = "failure_type"  # text label

# ---------------------------------------------------------------- features
# Model inhi columns par train hota hai. Order same rehna chahiye
# training aur prediction dono me, isliye artifact ke andar bhi save karte hain.
FEATURES = [
    "temperature", "air_temperature", "temp_diff",
    "rotational_speed", "torque", "tool_wear", "type_code",
    "power", "torque_speed_ratio",
    "temp_roll_mean", "temp_roll_std", "temp_delta", "temp_slope",
    "torque_roll_mean", "torque_delta",
    "speed_roll_std", "wear_rate",
]

ROLLING_WINDOW = 10      # rolling mean/std ke liye window
SLOPE_WINDOW = 10        # trend slope ke liye window
HISTORY_NEEDED = 12      # inference ke time DB se itni purani readings lenge

# ---------------------------------------------------------------- risk levels
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.50

RISK_ACTIONS = {
    "HIGH": "Inspect machine immediately",
    "MEDIUM": "Schedule inspection soon",
    "LOW": "Continue monitoring",
}

# Recommended action ke saath jo time-frame dikhta hai. Plant ki policy ke
# hisaab se badal sakte ho.
RISK_TIMEFRAME = {
    "HIGH": "within 24 hours",
    "MEDIUM": "within 7 days",
    "LOW": "",
}

# Risk level se maintenance priority
RISK_PRIORITY = {"HIGH": "Urgent", "MEDIUM": "Medium", "LOW": "Low"}
PRIORITIES = ["Urgent", "High", "Medium", "Low"]

# Failure type ke hisaab se kya check karna hai
FAILURE_ACTIONS = {
    "Heat Dissipation Failure": "Inspect cooling system and airflow",
    "Tool Wear Failure": "Inspect or replace the cutting tool",
    "Power Failure": "Check motor, power supply and drive",
    "Overstrain Failure": "Reduce load and inspect for mechanical overstrain",
    "Random Failure": "Carry out a general inspection",
}

# Maintenance form ke technician dropdown ki list (apni team ke naam daal do)
TECHNICIANS = ["Unassigned", "Technician 1", "Technician 2", "Technician 3"]

# ---------------------------------------------------------------- anomaly / RUL
# Anomaly score threshold se itna upar ho to level HIGH, warna MEDIUM
ANOMALY_HIGH_MARGIN = 0.05
# RUL (readings me) is value se kam ho to "maintenance recommended"
RUL_WARN = 10

# ---------------------------------------------------------------- demo data
# Synthetic dataset ki timeline January 2026 ki hai. True rakhne par database
# bharte waqt saari timestamps aise shift hoti hain ki sabse nayi reading
# "abhi" ki ho jaaye, taaki dashboard ke "Predictions Today" jaise cards live
# dikhein. Real data use kar rahe ho to False kar do.
SHIFT_SEED_TO_NOW = True
SEED_HISTORY_READINGS = 48   # har machine ki last itni readings par history banegi
SEED_STEP = 4                # inme se har 4th reading par prediction save hogi

# ---------------------------------------------------------------- training
TEST_FRACTION = 0.25     # time based split: last 25% time test ke liye
RANDOM_STATE = 42
RUL_CAP = 30             # RUL ko is value par cap karte hain (readings me)
MODEL_VERSION = "xgb_v1"

# ---------------------------------------------------------------- flask
MODEL_VERSION_LABEL = MODEL_VERSION
# Flash messages (jaise "record saved") session cookie use karte hain, isliye secret key chahiye.
# Environment variable SECRET_KEY set ho to wahi use hoti hai. Nahi to har run par nayi random
# key banti hai (demo ke liye theek hai; app restart par purane cookies invalid ho jaate hain).
SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
