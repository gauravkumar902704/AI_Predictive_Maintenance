"""
database.py
-----------
SQLite database layer (Python ka built-in sqlite3, koi extra install nahi).

Tables project documentation ke ERD ke hisaab se:
    machines, sensor_readings, failure_events,
    predictions, maintenance, ml_models, user_session

Baad me PostgreSQL use karna ho to sirf is file ke queries/connection
badalne padenge, baaki app waisa hi rahega.

Purani database file par bhi chalta hai: init_db() naye columns apne aap
add kar deta hai (_migrate), data delete nahi hota.
"""
import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.status import health_status  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    machine_id   TEXT PRIMARY KEY,
    machine_type TEXT,
    model_name   TEXT,
    install_date TEXT,
    status       TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id       TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    air_temperature  REAL,
    temperature      REAL,
    rotational_speed REAL,
    torque           REAL,
    tool_wear        REAL,
    FOREIGN KEY (machine_id) REFERENCES machines (machine_id)
);
CREATE INDEX IF NOT EXISTS idx_readings_machine_time
    ON sensor_readings (machine_id, timestamp);

CREATE TABLE IF NOT EXISTS failure_events (
    failure_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    failure_type TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines (machine_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id          TEXT NOT NULL,
    prediction_time     TEXT NOT NULL,
    failure_probability REAL,
    risk_level          TEXT,
    is_anomaly          INTEGER,
    anomaly_score       REAL,
    anomaly_level       TEXT,
    estimated_rul       REAL,
    likely_failure_type TEXT,
    top_factors         TEXT,
    model_version       TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines (machine_id)
);
CREATE INDEX IF NOT EXISTS idx_pred_machine_time
    ON predictions (machine_id, prediction_time);

CREATE TABLE IF NOT EXISTS maintenance (
    maintenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id     TEXT NOT NULL,
    date           TEXT NOT NULL,
    action         TEXT NOT NULL,
    technician     TEXT,
    issue          TEXT,
    priority       TEXT DEFAULT 'Medium',
    scheduled_date TEXT,
    status         TEXT DEFAULT 'Scheduled',
    notes          TEXT,
    created_at     TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines (machine_id)
);

CREATE TABLE IF NOT EXISTS ml_models (
    model_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    version    TEXT,
    algorithm  TEXT,
    metrics    TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_session (
    session_id TEXT PRIMARY KEY,
    user_id    TEXT,
    login_time TEXT
);
"""

# Purane version ki database me ye columns nahi the. init_db() inhe add kar deta hai.
NEW_COLUMNS = {
    "predictions": {
        "anomaly_score": "REAL",
        "anomaly_level": "TEXT",
    },
    "maintenance": {
        "issue": "TEXT",
        "priority": "TEXT DEFAULT 'Medium'",
        "scheduled_date": "TEXT",
        "status": "TEXT DEFAULT 'Scheduled'",
        "notes": "TEXT",
        "created_at": "TEXT",
    },
}

INSERT_PREDICTION = (
    "INSERT INTO predictions (machine_id, prediction_time, failure_probability, risk_level, "
    "is_anomaly, anomaly_score, anomaly_level, estimated_rul, likely_failure_type, "
    "top_factors, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

# "open" work order = jo abhi Completed nahi hua
OPEN_ORDER = "COALESCE(status, 'Scheduled') != 'Completed'"


# ------------------------------------------------------------------ connection
@contextmanager
def get_conn(db_path=None):
    """
    Connection deta hai: `with get_conn() as conn:` ki tarah use karo.
    Block khatam hone par commit hota hai (error aaye to rollback) aur
    connection band ho jaata hai. Band karna zaroori hai warna Windows par
    database file lock reh sakti hai.
    Rows ko dict ki tarah padh sakte hain (row_factory).
    """
    db_path = db_path or config.DB_PATH
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate(conn):
    """Purani database me missing columns add karta hai (data safe rehta hai)."""
    for table, columns in NEW_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def init_db(db_path=None):
    """Saare tables bana deta hai (agar pehle se hain to kuch nahi karta)."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _rows(cursor):
    return [dict(r) for r in cursor.fetchall()]


def _fix_anomaly_level(row):
    """Purani rows me anomaly_level nahi hota, isliye is_anomaly se bana dete hain."""
    if not row.get("anomaly_level"):
        row["anomaly_level"] = "MEDIUM" if row.get("is_anomaly") else "LOW"
    return row


# ------------------------------------------------------------------ seeding
def is_seeded(db_path=None):
    with get_conn(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
    return n > 0


def seed_from_dataframe(df, db_path=None):
    """
    Dataset ki rows ko database me daalta hai (sirf pehli baar).
    df me raw columns hone chahiye (machine_id, timestamp, sensors, failure...).
    """
    with get_conn(db_path) as conn:
        # machines
        first = df.groupby(config.COL_MACHINE).agg(
            mtype=(config.COL_TYPE, "first"),
            installed=(config.COL_TIME, "min"),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO machines (machine_id, machine_type, model_name, install_date, status) "
            "VALUES (?, ?, ?, ?, 'active')",
            [(mid, r.mtype, f"Model-{r.mtype}", str(r.installed.date())) for mid, r in first.iterrows()],
        )

        # sensor readings
        cols = [config.COL_MACHINE, config.COL_TIME, config.COL_AIR_TEMP, config.COL_TEMP,
                config.COL_SPEED, config.COL_TORQUE, config.COL_WEAR]
        data = df[cols].copy()
        data[config.COL_TIME] = data[config.COL_TIME].astype(str)
        conn.executemany(
            "INSERT INTO sensor_readings (machine_id, timestamp, air_temperature, temperature, "
            "rotational_speed, torque, tool_wear) VALUES (?, ?, ?, ?, ?, ?, ?)",
            data.itertuples(index=False, name=None),
        )

        # failure events
        fails = df[df[config.COL_FAILURE] == 1][[config.COL_MACHINE, config.COL_TIME, config.COL_FAILURE_TYPE]].copy()
        fails[config.COL_TIME] = fails[config.COL_TIME].astype(str)
        conn.executemany(
            "INSERT INTO failure_events (machine_id, timestamp, failure_type) VALUES (?, ?, ?)",
            fails.itertuples(index=False, name=None),
        )


def register_model(version, algorithm, metrics, db_path=None):
    """Model version ka record rakhta hai (ML_MODEL table)."""
    with get_conn(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM ml_models WHERE version = ? AND algorithm = ?", (version, algorithm)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO ml_models (version, algorithm, metrics) VALUES (?, ?, ?)",
                (version, algorithm, json.dumps(metrics)),
            )


def table_counts(db_path=None):
    """Settings page ke liye: har main table me kitni rows hain."""
    tables = ["machines", "sensor_readings", "failure_events", "predictions", "maintenance", "ml_models"]
    with get_conn(db_path) as conn:
        return {t: conn.execute("SELECT COUNT(*) FROM " + t).fetchone()[0] for t in tables}


# ------------------------------------------------------------------ machines
def machine_exists(machine_id, db_path=None):
    with get_conn(db_path) as conn:
        return conn.execute("SELECT 1 FROM machines WHERE machine_id = ?", (machine_id,)).fetchone() is not None


def add_machine(machine_id, machine_type="L", db_path=None):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO machines (machine_id, machine_type, model_name, install_date, status) "
            "VALUES (?, ?, ?, date('now'), 'active')",
            (machine_id, machine_type, f"Model-{machine_type}"),
        )


def get_machine(machine_id, db_path=None):
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT * FROM machines WHERE machine_id = ?", (machine_id,)).fetchone()
    return dict(row) if row else None


def list_machines(db_path=None):
    """
    Har machine ke saath uski latest prediction (agar hai), overall health
    aur open maintenance orders ki ginti. Zyada risk wali machine pehle.
    """
    sql = f"""
    SELECT m.machine_id, m.machine_type, m.install_date, m.status,
           p.prediction_time, p.failure_probability, p.risk_level,
           p.is_anomaly, p.anomaly_score, p.anomaly_level, p.estimated_rul,
           p.likely_failure_type,
           (SELECT COUNT(*) FROM maintenance mt
             WHERE mt.machine_id = m.machine_id
               AND COALESCE(mt.status, 'Scheduled') != 'Completed') AS open_orders
    FROM machines m
    LEFT JOIN predictions p ON p.prediction_id = (
        SELECT prediction_id FROM predictions
        WHERE machine_id = m.machine_id
        ORDER BY prediction_time DESC, prediction_id DESC LIMIT 1
    )
    ORDER BY COALESCE(p.failure_probability, -1) DESC, m.machine_id
    """
    with get_conn(db_path) as conn:
        machines = _rows(conn.execute(sql))
    for m in machines:
        if m["risk_level"]:
            _fix_anomaly_level(m)
            m["health"] = health_status(m["risk_level"], bool(m["is_anomaly"]))
        else:
            m["health"] = "UNKNOWN"
    return machines


# ------------------------------------------------------------------ readings
def get_recent_readings(machine_id, limit=None, db_path=None):
    """Machine ki pichli readings, purani se nayi order me (DataFrame)."""
    limit = limit or config.HISTORY_NEEDED
    sql = """
    SELECT machine_id, timestamp, air_temperature, temperature, rotational_speed,
           torque, tool_wear
    FROM (
        SELECT * FROM sensor_readings WHERE machine_id = ?
        ORDER BY timestamp DESC, reading_id DESC LIMIT ?
    ) ORDER BY timestamp ASC
    """
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(machine_id, limit))


def add_reading(machine_id, timestamp, reading, db_path=None):
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO sensor_readings (machine_id, timestamp, air_temperature, temperature, "
            "rotational_speed, torque, tool_wear) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (machine_id, str(timestamp), reading[config.COL_AIR_TEMP], reading[config.COL_TEMP],
             reading[config.COL_SPEED], reading[config.COL_TORQUE], reading[config.COL_WEAR]),
        )


def recent_readings_all(limit=500, db_path=None):
    """Drift check ke liye saari machines ki latest readings."""
    sql = ("SELECT temperature, rotational_speed, torque, tool_wear FROM sensor_readings "
           "ORDER BY timestamp DESC, reading_id DESC LIMIT ?")
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=(limit,))


def latest_readings_map(db_path=None):
    """
    Har machine ki sabse nayi reading (predict form ko auto-fill karne ke liye).
    Return: {machine_id: {machine_type, air_temperature, temperature, ...}}
    """
    sql = """
    SELECT s.machine_id, m.machine_type, s.air_temperature, s.temperature,
           s.rotational_speed, s.torque, s.tool_wear
    FROM sensor_readings s
    JOIN machines m ON m.machine_id = s.machine_id
    WHERE s.reading_id = (
        SELECT reading_id FROM sensor_readings
        WHERE machine_id = s.machine_id
        ORDER BY timestamp DESC, reading_id DESC LIMIT 1
    )
    """
    with get_conn(db_path) as conn:
        rows = _rows(conn.execute(sql))
    return {r["machine_id"]: r for r in rows}


# ------------------------------------------------------------------ predictions
def _prediction_values(machine_id, result, when):
    """Ek result dict ko insert ke tuple me badalta hai."""
    return (machine_id, str(when), result["failure_probability"], result.get("risk_level", "LOW"),
            int(result["is_anomaly"]), result.get("anomaly_score"), result.get("anomaly_level"),
            result["estimated_rul"], result["likely_failure_type"],
            json.dumps(result["top_factors"]), result["model_version"])


def add_prediction(machine_id, result, when, db_path=None):
    with get_conn(db_path) as conn:
        cur = conn.execute(INSERT_PREDICTION, _prediction_values(machine_id, result, when))
        return cur.lastrowid


def add_predictions_bulk(rows, db_path=None):
    """rows = [(machine_id, when, result_dict), ...]  (demo history ke liye)"""
    with get_conn(db_path) as conn:
        conn.executemany(INSERT_PREDICTION,
                         [_prediction_values(mid, res, when) for mid, when, res in rows])


def count_predictions(db_path=None):
    with get_conn(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]


def get_predictions(machine_id, limit=50, db_path=None):
    """Machine ki predictions, nayi pehle."""
    with get_conn(db_path) as conn:
        rows = _rows(conn.execute(
            "SELECT * FROM predictions WHERE machine_id = ? "
            "ORDER BY prediction_time DESC, prediction_id DESC LIMIT ?", (machine_id, limit)))
    for r in rows:
        r["top_factors"] = json.loads(r["top_factors"]) if r["top_factors"] else []
        _fix_anomaly_level(r)
    return rows


def get_recent_alerts(limit=10, db_path=None):
    """Latest HIGH risk predictions (dashboard ke alerts ke liye)."""
    with get_conn(db_path) as conn:
        return _rows(conn.execute(
            "SELECT machine_id, prediction_time, failure_probability, likely_failure_type "
            "FROM predictions WHERE risk_level = 'HIGH' "
            "ORDER BY prediction_time DESC, prediction_id DESC LIMIT ?", (limit,)))


def risk_summary(db_path=None):
    """Latest risk ke hisaab se kitni machines HIGH/MEDIUM/LOW hain."""
    summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    for m in list_machines(db_path):
        summary[m["risk_level"] or "UNKNOWN"] += 1
    return summary


def dashboard_summary(db_path=None):
    """
    Dashboard ke 6 cards ke numbers, seedha database se.
    maintenance_due = HIGH/MEDIUM risk wali machines jinka koi open work order nahi hai.
    """
    machines = list_machines(db_path)
    with get_conn(db_path) as conn:
        today = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE date(prediction_time) = date('now', 'localtime')"
        ).fetchone()[0]
        open_orders = conn.execute("SELECT COUNT(*) FROM maintenance WHERE " + OPEN_ORDER).fetchone()[0]
        completed = conn.execute("SELECT COUNT(*) FROM maintenance WHERE status = 'Completed'").fetchone()[0]

    return {
        "total_machines": len(machines),
        "healthy": sum(1 for m in machines if m["risk_level"] == "LOW"),
        "medium": sum(1 for m in machines if m["risk_level"] == "MEDIUM"),
        "high": sum(1 for m in machines if m["risk_level"] == "HIGH"),
        "unknown": sum(1 for m in machines if not m["risk_level"]),
        "anomalies": sum(1 for m in machines if m["is_anomaly"]),
        "predictions_today": today,
        "maintenance_due": sum(1 for m in machines
                               if m["risk_level"] in ("HIGH", "MEDIUM") and m["open_orders"] == 0),
        "open_orders": open_orders,
        "completed_orders": completed,
    }


def risk_trend(buckets=24, db_path=None):
    """
    Ghante ke hisaab se: us ghante ki predictions me kitne % MEDIUM/HIGH the.
    Percentage isliye ki kisi ghante me kam predictions hon to bhi chart na bigde.
    Return: purane se naye order me list of dict.
    """
    sql = """
    SELECT substr(prediction_time, 1, 13) AS bucket,
           COUNT(*) AS n,
           AVG(failure_probability) AS avg_probability,
           SUM(CASE WHEN risk_level IN ('MEDIUM', 'HIGH') THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS at_risk_percent
    FROM predictions
    GROUP BY bucket
    ORDER BY bucket DESC
    LIMIT ?
    """
    with get_conn(db_path) as conn:
        rows = _rows(conn.execute(sql, (buckets,)))
    rows.reverse()
    for r in rows:
        try:
            r["label"] = datetime.strptime(r["bucket"], "%Y-%m-%d %H").strftime("%d %b %H:00")
        except ValueError:
            r["label"] = r["bucket"]
    return rows


def failure_type_counts(db_path=None):
    """Past failure events ka type ke hisaab se count."""
    with get_conn(db_path) as conn:
        rows = _rows(conn.execute(
            "SELECT failure_type, COUNT(*) AS n FROM failure_events GROUP BY failure_type ORDER BY n DESC"))
    for r in rows:
        r["failure_type"] = r["failure_type"] or "Unknown"
    return rows


# ------------------------------------------------------------------ maintenance
def add_maintenance(machine_id, date, action, technician=None, issue=None, priority="Medium",
                    scheduled_date=None, notes=None, status="Scheduled", db_path=None):
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO maintenance (machine_id, date, action, technician, issue, priority, "
            "scheduled_date, status, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (machine_id, date, action, technician, issue, priority, scheduled_date, status, notes,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        return cur.lastrowid


def get_maintenance(machine_id, limit=20, db_path=None):
    with get_conn(db_path) as conn:
        return _rows(conn.execute(
            "SELECT * FROM maintenance WHERE machine_id = ? ORDER BY date DESC, maintenance_id DESC LIMIT ?",
            (machine_id, limit)))


def list_work_orders(status=None, limit=200, db_path=None):
    """
    Saare maintenance records. Open wale pehle (priority aur date ke hisaab se),
    completed baad me.  status = 'open' / 'completed' / None (sab).
    """
    where = ""
    if status == "open":
        where = "WHERE " + OPEN_ORDER
    elif status == "completed":
        where = "WHERE status = 'Completed'"
    sql = f"""
    SELECT * FROM maintenance {where}
    ORDER BY CASE WHEN status = 'Completed' THEN 1 ELSE 0 END,
             CASE priority WHEN 'Urgent' THEN 0 WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
             COALESCE(scheduled_date, date), maintenance_id DESC
    LIMIT ?
    """
    with get_conn(db_path) as conn:
        return _rows(conn.execute(sql, (limit,)))


def complete_work_order(order_id, db_path=None):
    """Work order ko Completed mark karta hai. Return: True agar record mila."""
    with get_conn(db_path) as conn:
        cur = conn.execute("UPDATE maintenance SET status = 'Completed' WHERE maintenance_id = ?", (order_id,))
        return cur.rowcount > 0


def get_failures(machine_id, limit=20, db_path=None):
    with get_conn(db_path) as conn:
        return _rows(conn.execute(
            "SELECT * FROM failure_events WHERE machine_id = ? ORDER BY timestamp DESC LIMIT ?",
            (machine_id, limit)))
