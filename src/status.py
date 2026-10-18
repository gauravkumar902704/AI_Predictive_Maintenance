"""
status.py
---------
Chhote pure functions: risk level, anomaly level, machine health aur
recommended action. Inme koi ML library nahi hai, isliye database aur
templates inhe bina heavy import ke use kar sakte hain.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def risk_from_probability(p):
    """Failure probability -> (risk level, short action)."""
    if p >= config.HIGH_THRESHOLD:
        level = "HIGH"
    elif p >= config.MEDIUM_THRESHOLD:
        level = "MEDIUM"
    else:
        level = "LOW"
    return level, config.RISK_ACTIONS[level]


def anomaly_level(score, is_anomaly, threshold):
    """
    Isolation Forest ka score -> LOW / MEDIUM / HIGH.
    threshold = model ka decision boundary (-iso.offset_). Boundary paar
    karte hi MEDIUM, aur margin se zyada door ho to HIGH.
    """
    if not is_anomaly:
        return "LOW"
    if score >= threshold + config.ANOMALY_HIGH_MARGIN:
        return "HIGH"
    return "MEDIUM"


def health_status(risk, is_anomaly):
    """Machine ki overall health: failure risk aur anomaly dono ko milakar."""
    if risk == "HIGH":
        return "CRITICAL"
    if risk == "MEDIUM" or is_anomaly:
        return "WARNING"
    return "HEALTHY"


def priority_for(risk):
    """Risk level se maintenance priority (Urgent / Medium / Low)."""
    return config.RISK_PRIORITY.get(risk, "Low")


def recommended_action(risk, failure_type=None):
    """
    Maintenance team ke liye ek line ka suggestion.
    Example: 'Inspect cooling system and airflow within 24 hours.'
    """
    if risk not in ("HIGH", "MEDIUM"):
        return "Continue monitoring"
    what = config.FAILURE_ACTIONS.get(failure_type, "Inspect the machine")
    return f"{what} {config.RISK_TIMEFRAME[risk]}."


def factor_summary(factors):
    """Top 2 factors ko ek sentence me: 'High tool wear and high temperature'."""
    parts = []
    for f in factors or []:
        name = f.get("name")
        if not name:
            continue
        parts.append(f"{f.get('direction', 'high')} {name.lower()}")
        if len(parts) == 2:
            break
    if not parts:
        return ""
    text = " and ".join(parts)
    return text[0].upper() + text[1:]
