"""
app.py
------
Flask application: dashboard, machines, prediction, analytics, anomalies,
maintenance, history, model performance, settings + JSON API.

Run:
    python app.py
Phir browser me kholo: http://127.0.0.1:5000/

Structure:
    create_app()      -> app banata hai (testing ke liye bhi use hota hai)
    run_prediction()  -> validate -> features -> model -> database me save
    page routes       -> HTML pages
    /api/...          -> JSON API
"""
import json
import os
from datetime import datetime

import joblib
import pandas as pd
import sklearn
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

import config
from src import charts
from src import database as db
from src.model_utils import predict_batch, predict_reading
from src.preprocessing import (FIELD_LABELS, build_training_frame,
                               features_for_new_reading, validate_reading)
from src.status import health_status, priority_for, recommended_action

# Chart colors (CSS variables, static/css/style.css me defined)
GREEN, YELLOW, RED, BLUE = "var(--green)", "var(--yellow)", "var(--red)", "var(--blue)"
CATEGORY_COLORS = ["var(--red)", "var(--yellow)", "var(--blue)", "var(--teal)", "var(--purple)", "var(--grey)"]

# Sidebar menu: (endpoint, label)
MENU = [
    ("dashboard", "Dashboard"),
    ("machines", "Machines"),
    ("predict", "Predict Health"),
    ("analytics", "Analytics"),
    ("anomalies", "Anomalies"),
    ("maintenance", "Maintenance"),
    ("history", "Prediction History"),
    ("model_performance", "Model Performance"),
    ("settings", "Settings"),
]

# Predict page par dikhne wale sensor inputs ka order
SENSOR_FIELDS = [config.COL_AIR_TEMP, config.COL_TEMP, config.COL_SPEED, config.COL_TORQUE, config.COL_WEAR]


# ====================================================================== helpers
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def short_time(value):
    """'2026-10-08 14:30:00' -> '08 Oct 14:30' (parse na ho to jaisa hai waisa)."""
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").strftime("%d %b %H:%M")
    except ValueError:
        return str(value) if value else "-"


def percent(value, decimals=1):
    """0.874 -> '87.4%'"""
    if value is None:
        return "-"
    return f"{float(value) * 100:.{decimals}f}%"


def load_metrics():
    """reports/metrics.json padhta hai (model ka score dikhane ke liye)."""
    try:
        with open(config.METRICS_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def safe_next(target, default):
    """Redirect sirf is site ke andar: '/maintenance' chalega, 'http://evil' ya '//evil' nahi."""
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return default


def extreme_value_warnings(artifact, cleaned):
    """
    Agar koi value training data ke normal range se bahut door hai
    (mean se 4 standard deviation se zyada) to warning dete hain.
    Prediction block nahi karte, sirf user ko batate hain.
    """
    warnings = []
    for col, st in artifact.get("feature_stats", {}).items():
        if col in cleaned and st["std"] > 0:
            if abs(cleaned[col] - st["mean"]) / st["std"] > 4:
                label = FIELD_LABELS.get(col, col).lower()
                warnings.append(f"The {label} value {cleaned[col]:g} is far outside the usual range "
                                f"(typical about {st['mean']:.1f}). Please check the sensor reading.")
    return warnings


def drift_report(app):
    """
    Simple drift indicator: live readings ka average training average se
    kitna hat gaya (training std ke unit me).
        < 0.5  -> stable, 0.5 - 1.0 -> watch, > 1.0 -> drift
    """
    stats = app.config["ARTIFACT"].get("feature_stats", {})
    live = db.recent_readings_all(500, app.config["DB_PATH"])
    report = []
    for col, st in stats.items():
        if len(live) == 0 or st["std"] == 0:
            continue
        shift = abs(float(live[col].mean()) - st["mean"]) / st["std"]
        status = "Stable" if shift < 0.5 else "Watch" if shift < 1.0 else "Drift"
        report.append({"feature": col, "label": FIELD_LABELS.get(col, col),
                       "train_mean": round(st["mean"], 2),
                       "live_mean": round(float(live[col].mean()), 2),
                       "shift": round(shift, 2), "status": status})
    return report


def trend_direction(values):
    """Sensor trend: pehle aur aakhri hisse ka average compare karke Rising / Falling / Stable."""
    vals = list(values)
    if len(vals) < 6:
        return "Stable"
    third = max(len(vals) // 3, 2)
    change = sum(vals[-third:]) / third - sum(vals[:third]) / third
    spread = float(pd.Series(vals).std()) or 1e-9
    if change > 0.5 * spread:
        return "Rising"
    if change < -0.5 * spread:
        return "Falling"
    return "Stable"


def sensor_cards(machine_id, db_path):
    """Machine page ke 4 sensor trend cards (sparkline + latest value + direction)."""
    df = db.get_recent_readings(machine_id, 60, db_path)
    specs = [(config.COL_TEMP, "Process temperature", "K"),
             (config.COL_TORQUE, "Torque", "Nm"),
             (config.COL_SPEED, "Rotational speed", "rpm"),
             (config.COL_WEAR, "Tool wear", "min")]
    cards = []
    for col, label, unit in specs:
        vals = df[col].tolist()
        cards.append({
            "label": label, "unit": unit,
            "latest": round(vals[-1], 1) if vals else None,
            "direction": trend_direction(vals),
            "chart": charts.sparkline(vals, color=BLUE),
        })
    return cards


def probability_chart(predictions, height=240):
    """Machine ki prediction history ka line chart (probability %, purane se naye)."""
    ordered = list(reversed(predictions))
    return charts.line_chart(
        [p["failure_probability"] * 100 for p in ordered],
        [short_time(p["prediction_time"]) for p in ordered],
        height=height, y_min=0, y_max=100, unit="%", color=BLUE,
        thresholds=[(config.MEDIUM_THRESHOLD * 100, "Medium", YELLOW),
                    (config.HIGH_THRESHOLD * 100, "High", RED)],
        aria="Failure probability history")


def wants_json():
    """API routes ko JSON error chahiye, pages ko HTML."""
    return request.path.startswith("/api/") or request.path == "/health"


# ====================================================================== bootstrap
def seed_predictions(artifact, df, path):
    """
    Demo history: har machine ki last N readings me se har SEED_STEP-vi par
    prediction nikal ke save karte hain (sabse nayi reading hamesha shamil).
    """
    recent = df.groupby(config.COL_MACHINE).tail(config.SEED_HISTORY_READINGS).copy()
    recent["_pos"] = recent.groupby(config.COL_MACHINE).cumcount(ascending=False)  # 0 = sabse nayi
    picked = recent[recent["_pos"] % config.SEED_STEP == 0]
    results = predict_batch(artifact, picked)
    rows = [(row[config.COL_MACHINE], str(row[config.COL_TIME]), res)
            for (_, row), res in zip(picked.iterrows(), results)]
    db.add_predictions_bulk(rows, path)
    return len(rows)


def bootstrap_database(app):
    """
    Pehli baar chalne par database bhar deta hai:
      1. tables banao (purani db ho to naye columns add karo)
      2. dataset ki readings/failures daalo
      3. har machine ki last readings par predictions save karo
         (taaki dashboard, history aur trend charts khali na dikhein)
    """
    path = app.config["DB_PATH"]
    artifact = app.config["ARTIFACT"]
    db.init_db(path)

    if not db.is_seeded(path):
        print("Preparing the database for the first time (this can take a little while)...")
        df = build_training_frame()
        if config.SHIFT_SEED_TO_NOW:
            # demo data ki saari timestamps aage khisakate hain: sabse nayi reading = abhi
            df[config.COL_TIME] = df[config.COL_TIME] + (pd.Timestamp.now().floor("min") - df[config.COL_TIME].max())
        db.seed_from_dataframe(df, path)
        n = seed_predictions(artifact, df, path)
        print(f"Database ready: {df[config.COL_MACHINE].nunique()} machines, {n} predictions.")

    metrics = load_metrics() or {}
    db.register_model(artifact["version"], artifact["algorithm"], metrics.get("primary", {}), path)


def run_prediction(app, machine_id, values):
    """
    Ek reading ka poora flow.
    Return: (result_dict, errors_list). Errors hone par result None hoga.
    """
    path = app.config["DB_PATH"]
    artifact = app.config["ARTIFACT"]

    machine_id = (machine_id or "").strip()
    cleaned, errors = validate_reading(values)
    if not machine_id:
        errors.append("Machine ID is required.")
    elif len(machine_id) > 30 or not all(c.isalnum() or c in "-_" for c in machine_id):
        errors.append("Machine ID may contain only letters, numbers, '-' and '_' (max 30 characters).")
    if errors:
        return None, errors

    # machine pehle se hai to uska saved type hi use hoga; nayi hai to register karo
    machine = db.get_machine(machine_id, path)
    if machine:
        cleaned[config.COL_TYPE] = machine["machine_type"] or cleaned[config.COL_TYPE]
    else:
        db.add_machine(machine_id, cleaned[config.COL_TYPE], path)

    # model ko us machine ki history chahiye (rolling features ke liye)
    history = db.get_recent_readings(machine_id, config.HISTORY_NEEDED, path)
    when = now_str()
    reading = {
        config.COL_MACHINE: machine_id,
        config.COL_TIME: when,
        config.COL_TYPE: cleaned[config.COL_TYPE],
        config.COL_AIR_TEMP: cleaned[config.COL_AIR_TEMP],
        config.COL_TEMP: cleaned[config.COL_TEMP],
        config.COL_SPEED: cleaned[config.COL_SPEED],
        config.COL_TORQUE: cleaned[config.COL_TORQUE],
        config.COL_WEAR: cleaned[config.COL_WEAR],
    }
    row = features_for_new_reading(history, reading)
    result = predict_reading(artifact, row)
    result["machine_id"] = machine_id
    result["timestamp"] = when
    result["warnings"] = extreme_value_warnings(artifact, cleaned)
    result["inputs"] = {k: reading[k] for k in SENSOR_FIELDS}

    db.add_reading(machine_id, when, reading, path)
    db.add_prediction(machine_id, result, when, path)
    return result, []


def validate_work_order(values, path):
    """
    Maintenance form / API ka data check karta hai.
    Return: (cleaned_dict, errors_list)
    """
    errors = []
    machine_id = (values.get("machine_id") or "").strip()
    if not machine_id:
        errors.append("Please select a machine.")
    elif not db.machine_exists(machine_id, path):
        errors.append(f"Machine '{machine_id}' was not found.")

    action = (values.get("action") or "").strip()
    if not action:
        errors.append("Action is required.")
    elif len(action) > 200:
        errors.append("Action must be 200 characters or fewer.")

    priority = (values.get("priority") or "Medium").strip()
    if priority not in config.PRIORITIES:
        errors.append("Priority must be one of: " + ", ".join(config.PRIORITIES) + ".")

    scheduled = (values.get("scheduled_date") or "").strip()
    if scheduled:
        try:
            datetime.strptime(scheduled, "%Y-%m-%d")
        except ValueError:
            errors.append("Scheduled date must be a valid date (YYYY-MM-DD).")

    notes = (values.get("notes") or "").strip()
    if len(notes) > 500:
        errors.append("Notes must be 500 characters or fewer.")

    technician = (values.get("technician") or "").strip()
    if technician in ("", "Unassigned"):
        technician = None
    elif len(technician) > 60:
        errors.append("Technician name must be 60 characters or fewer.")

    cleaned = {
        "machine_id": machine_id,
        "action": action,
        "issue": (values.get("issue") or "").strip()[:100] or None,
        "priority": priority,
        "scheduled_date": scheduled or None,
        "notes": notes or None,
        "technician": technician,
    }
    return cleaned, errors


def load_artifact(model_path):
    """
    Model file padhta hai. Agar wo alag scikit-learn version se bani hai to error deta hai:
    pickle aisi file par chal to jaati hai par predictions bigad sakti hain.
    """
    artifact = joblib.load(model_path)
    saved = artifact.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        raise RuntimeError(f"it was trained with scikit-learn {saved} but {sklearn.__version__} is installed")
    return artifact


# ====================================================================== app factory
def create_app(db_path=None, model_path=None, auto_train=True):
    """
    db_path / model_path testing me alag dene ke liye hain.
    auto_train=True: model file na mile to pehle train kar leta hai.
    """
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["DB_PATH"] = db_path or config.DB_PATH
    model_path = model_path or config.MODEL_PATH
    app.config["MODEL_ERROR"] = None
    app.config["ARTIFACT"] = None

    # ---------------------------------------------------------- model load
    if not os.path.exists(model_path) and auto_train:
        print("Model file not found, training now...")
        from src import data_generator, train
        if not os.path.exists(config.RAW_DATA_PATH):
            data_generator.main()
        train.main()

    if not os.path.exists(model_path):
        app.config["MODEL_ERROR"] = (
            f"Model file not found: {model_path}. Run 'python -m src.train' first."
        )
    else:
        try:
            try:
                app.config["ARTIFACT"] = load_artifact(model_path)
            except Exception as exc:  # corrupt file, version mismatch, etc.
                # Default model file ho to naye versions ke saath khud dobara train kar lete hain
                if not (auto_train and model_path == config.MODEL_PATH):
                    raise
                print(f"The saved model could not be loaded ({exc}). Retraining with the installed libraries...")
                from src import train
                train.main()
                app.config["ARTIFACT"] = load_artifact(model_path)
            bootstrap_database(app)
        except Exception as exc:
            app.config["ARTIFACT"] = None
            app.config["MODEL_ERROR"] = (
                f"The model could not be loaded ({exc}). Run 'python -m src.train' again."
            )

    app.add_template_filter(short_time, "shorttime")
    app.add_template_filter(percent, "percent")

    # ---------------------------------------------------------- layout / guards
    @app.context_processor
    def inject_layout():
        """Har template ko sidebar menu aur system status mil jaata hai."""
        artifact = app.config["ARTIFACT"]
        ok = app.config["MODEL_ERROR"] is None and artifact is not None
        return {
            "menu": MENU,
            "system_ok": ok,
            "model_label": f"{artifact['algorithm']} ({artifact['version']})" if ok else "No model loaded",
        }

    @app.before_request
    def check_model():
        # model na ho to health ke alawa sab par clear error
        if app.config["MODEL_ERROR"] and request.path != "/health" \
                and not request.path.startswith("/static"):
            if wants_json():
                return jsonify(error=app.config["MODEL_ERROR"]), 503
            return render_template("error.html", code=503, message=app.config["MODEL_ERROR"]), 503

    @app.errorhandler(404)
    def not_found(_e):
        if wants_json():
            return jsonify(error="Not found"), 404
        return render_template("error.html", code=404,
                               message="The page or machine you asked for was not found."), 404

    @app.errorhandler(500)
    def server_error(_e):
        if wants_json():
            return jsonify(error="Internal server error"), 500
        return render_template("error.html", code=500,
                               message="Something went wrong. Please check the server logs."), 500

    # ================================================================ pages
    @app.route("/")
    def dashboard():
        path = app.config["DB_PATH"]
        summary = db.dashboard_summary(path)
        machines = db.list_machines(path)

        # risk trend (ghante ke hisaab se at-risk predictions ka %)
        trend = db.risk_trend(24, path)
        trend_chart = charts.line_chart(
            [t["at_risk_percent"] for t in trend], [t["label"] for t in trend],
            y_min=0, unit="%", color=RED, aria="Share of predictions at medium or high risk")

        health_donut = charts.donut(
            [("Healthy", summary["healthy"], GREEN),
             ("Medium risk", summary["medium"], YELLOW),
             ("High risk", summary["high"], RED)], center_label="machines")

        # kin machines par dhyan chahiye (HIGH pehle, phir MEDIUM; list already probability se sorted)
        attention = [m for m in machines if m["risk_level"] in ("HIGH", "MEDIUM")]
        for m in attention:
            m["action"] = recommended_action(m["risk_level"], m["likely_failure_type"])

        drift = drift_report(app)
        drift_status = ("Drift" if any(d["status"] == "Drift" for d in drift)
                        else "Watch" if any(d["status"] == "Watch" for d in drift) else "Stable")

        return render_template(
            "dashboard.html", summary=summary, trend_chart=trend_chart, health_donut=health_donut,
            attention=attention[:8], attention_total=len(attention),
            alerts=db.get_recent_alerts(6, path), drift_status=drift_status)

    @app.route("/machines")
    def machines():
        path = app.config["DB_PATH"]
        risk = request.args.get("risk", "").upper()
        anomaly_only = request.args.get("anomaly") == "1"
        query = (request.args.get("q") or "").strip().lower()

        rows = db.list_machines(path)
        if risk in ("HIGH", "MEDIUM", "LOW"):
            rows = [m for m in rows if m["risk_level"] == risk]
        if anomaly_only:
            rows = [m for m in rows if m["is_anomaly"]]
        if query:
            rows = [m for m in rows if query in m["machine_id"].lower()]
        return render_template("machines.html", machines=rows, risk=risk, anomaly_only=anomaly_only,
                               q=request.args.get("q", ""), total=len(db.list_machines(path)))

    @app.route("/machine/<machine_id>")
    def machine_page(machine_id):
        path = app.config["DB_PATH"]
        machine = db.get_machine(machine_id, path)
        if not machine:
            return render_template("error.html", code=404, message=f"Machine '{machine_id}' was not found."), 404

        predictions = db.get_predictions(machine_id, 15, path)
        latest = predictions[0] if predictions else None
        extra = {}
        if latest:
            rul = latest["estimated_rul"] or 0
            extra = {
                "rul_percent": round(min(rul / config.RUL_CAP, 1) * 100, 1),
                "rul_status": "No immediate action" if rul > config.RUL_WARN else "Maintenance recommended",
                "action": recommended_action(latest["risk_level"], latest["likely_failure_type"]),
                "priority": priority_for(latest["risk_level"]),
                "health": health_status(latest["risk_level"], bool(latest["is_anomaly"])),
            }
        return render_template(
            "machine.html", machine=machine, latest=latest, extra=extra, predictions=predictions,
            sensors=sensor_cards(machine_id, path), chart=probability_chart(predictions) if predictions else None,
            orders=db.get_maintenance(machine_id, 10, path), failures=db.get_failures(machine_id, 10, path))

    def render_predict(form, errors=(), status=200):
        path = app.config["DB_PATH"]
        return render_template(
            "predict.html", machines=[m["machine_id"] for m in db.list_machines(path)],
            latest=db.latest_readings_map(path), form=form, errors=list(errors)), status

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        if request.method == "GET":
            # ?machine=M001 se form us machine ki latest reading se bhar jaata hai
            form = {"machine_id": request.args.get("machine", "")}
            form.update(db.latest_readings_map(app.config["DB_PATH"]).get(form["machine_id"], {}))
            return render_predict(form)

        machine_id = request.form.get("machine_id", "")
        if machine_id == "__new__":
            machine_id = request.form.get("new_machine_id", "")
        result, errors = run_prediction(app, machine_id, request.form)
        if errors:
            return render_predict(request.form, errors, 400)
        return render_template("result.html", r=result)

    @app.route("/analytics")
    def analytics():
        path = app.config["DB_PATH"]
        summary = db.dashboard_summary(path)
        machines = db.list_machines(path)

        types = db.failure_type_counts(path)
        failure_donut = charts.donut(
            [(t["failure_type"], t["n"], CATEGORY_COLORS[i % len(CATEGORY_COLORS)]) for i, t in enumerate(types)],
            center_label="failures")

        risk_bars = charts.vbar([("Healthy", summary["healthy"], GREEN),
                                 ("Medium", summary["medium"], YELLOW),
                                 ("High", summary["high"], RED)])

        trend = db.risk_trend(24, path)
        trend_chart = charts.line_chart(
            [t["at_risk_percent"] for t in trend], [t["label"] for t in trend],
            y_min=0, unit="%", color=RED, aria="Share of predictions at medium or high risk")

        # at-risk machines me kaun sa issue sabse zyada expected hai
        issues = {}
        for m in machines:
            if m["risk_level"] in ("HIGH", "MEDIUM") and m["likely_failure_type"]:
                issues[m["likely_failure_type"]] = issues.get(m["likely_failure_type"], 0) + 1
        issue_bars = charts.hbar(sorted(issues.items(), key=lambda kv: kv[1], reverse=True), color=BLUE)

        return render_template("analytics.html", summary=summary, failure_donut=failure_donut,
                               risk_bars=risk_bars, trend_chart=trend_chart, issue_bars=issue_bars,
                               total_failures=sum(t["n"] for t in types))

    @app.route("/anomalies")
    def anomalies():
        path = app.config["DB_PATH"]
        flagged = [m for m in db.list_machines(path) if m["is_anomaly"]]
        flagged.sort(key=lambda m: m["anomaly_score"] or 0, reverse=True)
        return render_template("anomalies.html", machines=flagged, metrics=load_metrics(),
                               total=len(db.list_machines(path)))

    def render_maintenance(form=None, errors=(), status=200):
        path = app.config["DB_PATH"]
        machines = db.list_machines(path)
        attention = [m for m in machines if m["risk_level"] in ("HIGH", "MEDIUM")]
        for m in attention:
            m["action"] = recommended_action(m["risk_level"], m["likely_failure_type"])

        if form is None:
            # ?machine=M023 se form machine ke latest prediction se pre-fill hota hai
            form = {"scheduled_date": datetime.now().strftime("%Y-%m-%d"), "priority": "Medium"}
            chosen = next((m for m in machines if m["machine_id"] == request.args.get("machine")), None)
            if chosen:
                form.update({
                    "machine_id": chosen["machine_id"],
                    "issue": chosen["likely_failure_type"] if chosen["risk_level"] in ("HIGH", "MEDIUM") else "",
                    "priority": priority_for(chosen["risk_level"]) if chosen["risk_level"] else "Medium",
                    "action": recommended_action(chosen["risk_level"], chosen["likely_failure_type"])
                    if chosen["risk_level"] in ("HIGH", "MEDIUM") else "",
                })
        flt = request.args.get("status", "")
        return render_template(
            "maintenance.html", machines=machines, attention=attention, form=form, errors=list(errors),
            orders=db.list_work_orders(flt if flt in ("open", "completed") else None, 200, path),
            status_filter=flt, priorities=config.PRIORITIES, technicians=config.TECHNICIANS,
            summary=db.dashboard_summary(path)), status

    @app.route("/maintenance", methods=["GET", "POST"])
    def maintenance():
        path = app.config["DB_PATH"]
        if request.method == "GET":
            return render_maintenance()

        cleaned, errors = validate_work_order(request.form, path)
        if errors:
            return render_maintenance(request.form, errors, 400)
        db.add_maintenance(cleaned["machine_id"], datetime.now().strftime("%Y-%m-%d"), cleaned["action"],
                           cleaned["technician"], cleaned["issue"], cleaned["priority"],
                           cleaned["scheduled_date"], cleaned["notes"], db_path=path)
        flash(f"Maintenance record saved for {cleaned['machine_id']}.", "success")
        return redirect(safe_next(request.form.get("next"), url_for("maintenance")))

    @app.route("/maintenance/<int:order_id>/complete", methods=["POST"])
    def maintenance_complete(order_id):
        if not db.complete_work_order(order_id, app.config["DB_PATH"]):
            return render_template("error.html", code=404, message="Maintenance record not found."), 404
        flash("Maintenance record marked as completed.", "success")
        return redirect(safe_next(request.form.get("next"), url_for("maintenance")))

    @app.route("/history")
    def history():
        path = app.config["DB_PATH"]
        machines = db.list_machines(path)
        machine_id = request.args.get("machine") or next(
            (m["machine_id"] for m in machines if m["risk_level"]), "")
        if not machine_id or not db.machine_exists(machine_id, path):
            return render_template("error.html", code=404, message=f"Machine '{machine_id}' was not found."), 404
        predictions = db.get_predictions(machine_id, 60, path)
        return render_template("history.html", machines=[m["machine_id"] for m in machines],
                               machine_id=machine_id, predictions=predictions,
                               chart=probability_chart(predictions, 280))

    @app.route("/model")
    def model_performance():
        available = set(os.listdir(config.FIGURES_DIR)) if os.path.isdir(config.FIGURES_DIR) else set()
        wanted = [("08_confusion_matrix.png", "Confusion matrix"),
                  ("09_roc_pr_curves.png", "ROC and precision-recall curves"),
                  ("10_feature_importance.png", "Feature importance")]
        return render_template("model_performance.html", metrics=load_metrics(),
                               artifact=app.config["ARTIFACT"],
                               figures=[(f, t) for f, t in wanted if f in available])

    @app.route("/figures/<path:filename>")
    def figures(filename):
        """Training ke plots (reports/figures) serve karta hai."""
        return send_from_directory(config.FIGURES_DIR, filename)

    @app.route("/settings")
    def settings():
        path = app.config["DB_PATH"]
        return render_template(
            "settings.html", artifact=app.config["ARTIFACT"], counts=db.table_counts(path),
            drift=drift_report(app), db_path=path, config=config)

    # ================================================================ API
    @app.route("/health")
    def health():
        ok = app.config["MODEL_ERROR"] is None
        body = {"status": "ok" if ok else "error",
                "model_loaded": ok,
                "model_version": app.config["ARTIFACT"]["version"] if ok else None,
                "algorithm": app.config["ARTIFACT"]["algorithm"] if ok else None}
        if not ok:
            body["error"] = app.config["MODEL_ERROR"]
        return jsonify(body), (200 if ok else 503)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="A JSON body is required."), 400
        result, errors = run_prediction(app, data.get("machine_id"), data)
        if errors:
            return jsonify(errors=errors), 400
        return jsonify(result)

    @app.route("/api/summary")
    def api_summary():
        return jsonify(db.dashboard_summary(app.config["DB_PATH"]))

    @app.route("/api/machines")
    def api_machines():
        return jsonify(db.list_machines(app.config["DB_PATH"]))

    @app.route("/api/machine/<machine_id>")
    def api_machine(machine_id):
        path = app.config["DB_PATH"]
        machine = db.get_machine(machine_id, path)
        if not machine:
            return jsonify(error=f"Machine '{machine_id}' not found"), 404
        preds = db.get_predictions(machine_id, 1, path)
        machine["latest_prediction"] = preds[0] if preds else None
        machine["maintenance"] = db.get_maintenance(machine_id, 5, path)
        machine["failures"] = db.get_failures(machine_id, 5, path)
        return jsonify(machine)

    @app.route("/api/history/<machine_id>")
    def api_history(machine_id):
        path = app.config["DB_PATH"]
        if not db.machine_exists(machine_id, path):
            return jsonify(error=f"Machine '{machine_id}' not found"), 404
        limit = max(1, min(request.args.get("limit", default=50, type=int), 500))
        return jsonify(db.get_predictions(machine_id, limit, path))

    @app.route("/api/maintenance", methods=["GET", "POST"])
    def api_maintenance():
        path = app.config["DB_PATH"]
        if request.method == "GET":
            status = request.args.get("status")
            return jsonify(db.list_work_orders(status if status in ("open", "completed") else None, 200, path))

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="A JSON body is required."), 400
        machine_id = (data.get("machine_id") or "").strip()
        if not machine_id or not (data.get("action") or "").strip():
            return jsonify(error="machine_id and action are required."), 400
        if not db.machine_exists(machine_id, path):
            return jsonify(error=f"Machine '{machine_id}' not found"), 404
        cleaned, errors = validate_work_order(data, path)
        if errors:
            return jsonify(errors=errors), 400
        new_id = db.add_maintenance(
            cleaned["machine_id"], data.get("date") or datetime.now().strftime("%Y-%m-%d"), cleaned["action"],
            cleaned["technician"], cleaned["issue"], cleaned["priority"], cleaned["scheduled_date"],
            cleaned["notes"], db_path=path)
        return jsonify(maintenance_id=new_id, status="saved"), 201

    @app.route("/api/drift")
    def api_drift():
        return jsonify(drift_report(app))

    return app


if __name__ == "__main__":
    # debug=False rakha hai (production jaisa). Development me badalna ho to
    # environment variable FLASK_DEBUG=1 set karo.
    application = create_app()
    application.run(host="127.0.0.1", port=5000,
                    debug=os.environ.get("FLASK_DEBUG") == "1")
