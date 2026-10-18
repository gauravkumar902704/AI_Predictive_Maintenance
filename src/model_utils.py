"""
model_utils.py
--------------
Model se related helper functions:
  - boosted classifier banana (XGBoost, na mile to sklearn fallback)
  - prediction ka explanation (kaun se sensor groups risk badha rahe hain)
  - ek reading (ya bahut saari readings) ke liye poora prediction result

Risk level, health status, recommended action jaise chhote rules status.py me hain.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.status import (anomaly_level, factor_summary, health_status,  # noqa: E402
                        priority_for, recommended_action,
                        risk_from_probability)

# XGBoost optional rakha hai. Install hai to wahi use hoga, warna
# sklearn ka HistGradientBoosting (same family ka algorithm) chalega.
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:  # ImportError ya binary problem
    HAS_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingClassifier

# Explanation me 17 alag features dikhane ke bajay unhe sensor groups me baant diya hai,
# taaki user ko "Tool wear", "Torque" jaise naam dikhein.
#   (naam, group ke features, "typical value" dikhane wala feature, unit)
EXPLAIN_GROUPS = [
    ("Tool wear", ["tool_wear", "wear_rate"], "tool_wear", "min"),
    ("Process - air temperature", ["temp_diff"], "temp_diff", "K"),
    ("Process temperature", ["temperature", "air_temperature", "temp_roll_mean", "temp_roll_std"],
     "temperature", "K"),
    ("Temperature slope", ["temp_delta", "temp_slope"], "temp_slope", "K/reading"),
    ("Torque", ["torque", "torque_roll_mean", "torque_delta"], "torque", "Nm"),
    ("Rotational speed", ["rotational_speed", "speed_roll_std"], "rotational_speed", "rpm"),
    ("Mechanical power", ["power", "torque_speed_ratio"], "power", "W"),
    ("Machine type", ["type_code"], "type_code", ""),
]

# Group ka asar itna kam ho to usse "risk driver" nahi maante (log-odds me)
MIN_DRIVER_EFFECT = 0.05


# ====================================================================== model
def make_boosted_classifier(pos_weight=1.0):
    """
    Primary model banata hai. Return: (model, algorithm_name)
    pos_weight = (healthy rows / failure rows) taaki imbalance handle ho.
    """
    if HAS_XGBOOST:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight,
            random_state=config.RANDOM_STATE,
        )
        return model, "XGBoost"

    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=5,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )
    return model, "HistGradientBoosting (sklearn fallback)"


# ====================================================================== explanation
def _logit(p):
    """Probability ko log-odds me badalta hai (0 aur 1 par clip karke)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _margin(model, X):
    """
    Model ka raw score (log-odds). Probability 100% ke paas saturate ho jaati hai
    (0.9999 aur 1.0 me farak dikhna band), isliye jahan model seedha raw score
    deta hai wahi use karte hain.
    """
    if HAS_XGBOOST and isinstance(model, XGBClassifier):
        try:
            return np.asarray(model.predict(X, output_margin=True), dtype=float)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return _logit(model.predict_proba(X)[:, 1])


def explain_prediction(artifact, row, top_n=5):
    """
    Local explanation (SHAP jaisa simple version, bina extra library ke).

    Har sensor group ko ek ek karke training-median se replace karte hain aur
    dekhte hain ki failure log-odds kitne gir jaate hain. Jis group ko hatane
    par risk sabse zyada girta hai wahi risk badha raha tha. Probability ki
    jagah log-odds isliye liye ki 100% ke paas probability saturate ho jaati hai.

    Return: list of dict, share = us group ka hissa (%) total risk drivers me.
    """
    model = artifact["model"]
    features = artifact["features"]
    baseline = artifact["baseline"]

    x = row[features].astype(float).reset_index(drop=True)
    # variants[0] = asli row, variants[i] = i-th group typical value par
    variants = pd.concat([x] * (len(EXPLAIN_GROUPS) + 1), ignore_index=True)
    for i, (_name, cols, _measure, _unit) in enumerate(EXPLAIN_GROUPS, start=1):
        for c in cols:
            if c in baseline:
                variants.loc[i, c] = baseline[c]

    margins = _margin(model, variants)
    drops = margins[0] - margins[1:]

    drivers = []
    for (name, _cols, measure, unit), drop in zip(EXPLAIN_GROUPS, drops):
        if drop > MIN_DRIVER_EFFECT:
            value = float(x.iloc[0][measure])
            typical = float(baseline[measure])
            drivers.append({
                "name": name,
                "effect": float(drop),
                "measure": measure,
                "unit": unit,
                "value": round(value, 2),
                "typical": round(typical, 2),
                "direction": "high" if value > typical else "low",
            })

    total = sum(d["effect"] for d in drivers)
    for d in drivers:
        d["share"] = round(d.pop("effect") / total * 100, 1)
    drivers.sort(key=lambda d: d["share"], reverse=True)
    return drivers[:top_n]


# ====================================================================== result
def _assemble(artifact, prob, is_anomaly, anomaly_score, type_probs, rul, factors):
    """Alag alag model outputs ko ek result dictionary me jodta hai."""
    risk, action = risk_from_probability(prob)

    # --- failure type: "No Failure" ko chhodkar sabse likely type
    classes = list(artifact["failure_type_model"].classes_)
    fail_idx = [i for i, c in enumerate(classes) if c != "No Failure"]
    if fail_idx:
        best = max(fail_idx, key=lambda i: type_probs[i])
        likely_type, likely_prob = classes[best], float(type_probs[best])
    else:
        likely_type, likely_prob = "Unknown", 0.0

    # --- anomaly: Isolation Forest ka decision boundary = -offset_
    threshold = -float(artifact["anomaly_model"].offset_)
    a_level = anomaly_level(anomaly_score, is_anomaly, threshold)

    # --- remaining useful life (readings me)
    rul = float(np.clip(rul, 0, config.RUL_CAP))
    rul_ok = rul > config.RUL_WARN

    # --- risk aur anomaly ko milakar interpretation
    if risk in ("HIGH", "MEDIUM") and is_anomaly:
        note = "Strong candidate for immediate inspection: high risk and abnormal behaviour together."
    elif risk == "LOW" and is_anomaly:
        note = "Failure risk is low but the readings look unusual. Possible sensor issue or an unknown fault."
    elif risk in ("HIGH", "MEDIUM"):
        note = "Failure risk is elevated although the sensor pattern is within the usual range."
    else:
        note = "Machine looks normal."

    return {
        "failure_probability": round(float(prob), 4),
        "probability_percent": round(float(prob) * 100, 1),
        "risk_level": risk,
        "action": action,
        "recommended_action": recommended_action(risk, likely_type),
        "priority": priority_for(risk),
        "health_status": health_status(risk, bool(is_anomaly)),
        "predicted_class": "Failure likely" if prob >= config.MEDIUM_THRESHOLD else "No failure expected",
        "is_anomaly": bool(is_anomaly),
        "anomaly_score": round(float(anomaly_score), 4),
        "anomaly_level": a_level,
        "likely_failure_type": likely_type,
        "likely_failure_type_prob": round(likely_prob, 4),
        "estimated_rul": round(rul, 1),
        "rul_percent": round(rul / config.RUL_CAP * 100, 1),
        "rul_status": "No immediate action" if rul_ok else "Maintenance recommended",
        "interpretation": note,
        "top_factors": factors,
        "factor_summary": factor_summary(factors),
        "model_version": artifact["version"],
        "algorithm": artifact["algorithm"],
    }


def _run_models(artifact, X):
    """Saare models ek saath chalata hai. Return: probs, anomaly flags/scores, type probs, rul."""
    probs = artifact["model"].predict_proba(X)[:, 1]
    iso = artifact["anomaly_model"]
    flags = iso.predict(X) == -1
    scores = -iso.score_samples(X)          # bada = zyada abnormal
    type_probs = artifact["failure_type_model"].predict_proba(X)
    rul = artifact["rul_model"].predict(X)
    return probs, flags, scores, type_probs, rul


def predict_reading(artifact, feature_row):
    """
    Ek feature row (DataFrame, 1 row) par saare models chalata hai, explanation ke saath.
    Return: dictionary jo Flask template/API dono me use hoti hai.
    """
    X = feature_row[artifact["features"]].astype(float)
    probs, flags, scores, type_probs, rul = _run_models(artifact, X)
    factors = explain_prediction(artifact, feature_row) if probs[0] >= config.MEDIUM_THRESHOLD else []
    return _assemble(artifact, probs[0], flags[0], scores[0], type_probs[0], rul[0], factors)


def predict_batch(artifact, df):
    """
    Bahut saari rows ek saath (demo history banane ke liye). Explanation sirf
    MEDIUM/HIGH rows ke liye nikalte hain (baaki me explain karne ko kuch hai nahi),
    isliye LOW rows me top_factors khali list hogi.
    Return: list of result dicts, df ki row order me.
    """
    X = df[artifact["features"]].astype(float)
    probs, flags, scores, type_probs, rul = _run_models(artifact, X)
    results = []
    for i in range(len(X)):
        factors = []
        if probs[i] >= config.MEDIUM_THRESHOLD:
            factors = explain_prediction(artifact, df.iloc[[i]])
        results.append(_assemble(artifact, probs[i], flags[i], scores[i], type_probs[i], rul[i], factors))
    return results
