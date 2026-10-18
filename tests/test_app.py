"""
test_app.py
-----------
Project documentation ke "Testing Checklist" ke hisaab se tests.

Run:
    python -m unittest discover -s tests -v

Tests ek temporary database use karte hain, isliye asli data/maintenance.db
par koi asar nahi padta. Model file pehle se honi chahiye (python -m src.train).
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from app import create_app  # noqa: E402
from src import charts  # noqa: E402
from src.preprocessing import add_features, clean, validate_reading  # noqa: E402
from src.status import (anomaly_level, health_status, priority_for,  # noqa: E402
                        recommended_action, risk_from_probability)

# Ek normal healthy reading aur ek aisi reading jisme failure ke signs hain
NORMAL = {"machine_id": "T-NORMAL", "machine_type": "M", "air_temperature": 300.0,
          "temperature": 309.6, "rotational_speed": 1500, "torque": 40, "tool_wear": 30}
RISKY = {"machine_id": "T-RISKY", "machine_type": "L", "air_temperature": 300.0,
         "temperature": 308.0, "rotational_speed": 1250, "torque": 55, "tool_wear": 230}

PAGES = ["/", "/machines", "/machine/M001", "/predict", "/analytics", "/anomalies",
         "/maintenance", "/history", "/history?machine=M001", "/model", "/settings"]


def second_machine_id(client):
    return client.get("/api/machines").get_json()[1]["machine_id"]


class AppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = os.path.join(cls.tmp.name, "test.db")
        if not os.path.exists(config.MODEL_PATH):
            raise unittest.SkipTest("Model file nahi hai. Pehle 'python -m src.train' chalao.")
        cls.app = create_app(db_path=cls.db_path, auto_train=False)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------------------------------------------------------- input validation
    def test_missing_sensor_gives_validation_error(self):
        data = dict(NORMAL)
        del data["torque"]
        r = self.client.post("/api/predict", json=data)
        self.assertEqual(r.status_code, 400)
        self.assertTrue(any("Torque" in e for e in r.get_json()["errors"]))

    def test_non_numeric_input_gives_friendly_error(self):
        r = self.client.post("/api/predict", json=dict(NORMAL, torque="abc"))
        self.assertEqual(r.status_code, 400)

    def test_form_non_numeric_shows_error_on_the_form(self):
        r = self.client.post("/predict", data=dict(NORMAL, temperature="hot"))
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"must be a number", r.data)
        self.assertIn(b'id="predict-form"', r.data)   # form wapas dikhta hai

    def test_impossible_value_rejected(self):
        r = self.client.post("/api/predict", json=dict(NORMAL, rotational_speed=-50))
        self.assertEqual(r.status_code, 400)

    def test_bad_machine_id_rejected(self):
        r = self.client.post("/api/predict", json=dict(NORMAL, machine_id="<script>"))
        self.assertEqual(r.status_code, 400)

    def test_extreme_value_gives_warning(self):
        r = self.client.post("/api/predict", json=dict(NORMAL, machine_id="T-EXTREME", tool_wear=900))
        self.assertEqual(r.status_code, 200)
        self.assertTrue(len(r.get_json()["warnings"]) > 0)

    # ---------------------------------------------------------- prediction
    def test_normal_sample_is_low_risk(self):
        body = self.client.post("/api/predict", json=NORMAL).get_json()
        self.assertEqual(body["risk_level"], "LOW")
        self.assertEqual(body["health_status"], "HEALTHY")

    def test_known_failure_sample_is_high_risk_with_explanation(self):
        body = self.client.post("/api/predict", json=RISKY).get_json()
        self.assertGreater(body["failure_probability"], 0.5)
        self.assertIn(body["risk_level"], ("MEDIUM", "HIGH"))
        self.assertEqual(body["health_status"] if body["risk_level"] == "HIGH" else "CRITICAL", "CRITICAL")
        self.assertTrue(body["top_factors"], "risky reading ke liye explanation honi chahiye")
        shares = [f["share"] for f in body["top_factors"]]
        self.assertEqual(shares, sorted(shares, reverse=True))
        self.assertTrue(all(0 < s <= 100 for s in shares))
        self.assertTrue(body["factor_summary"])
        self.assertIn("within", body["recommended_action"])

    def test_prediction_output_has_expected_fields(self):
        body = self.client.post("/api/predict", json=dict(NORMAL, machine_id="T-FIELDS")).get_json()
        for key in ("failure_probability", "risk_level", "recommended_action", "priority", "health_status",
                    "is_anomaly", "anomaly_level", "likely_failure_type", "estimated_rul", "rul_status",
                    "top_factors", "model_version"):
            self.assertIn(key, body)
        self.assertTrue(0.0 <= body["failure_probability"] <= 1.0)

    def test_prediction_is_saved_in_history(self):
        self.client.post("/api/predict", json=dict(NORMAL, machine_id="T-HIST"))
        self.client.post("/api/predict", json=dict(NORMAL, machine_id="T-HIST"))
        r = self.client.get("/api/history/T-HIST")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()), 2)

    def test_existing_machine_keeps_its_saved_type(self):
        # M001 ka type DB me hai; form me galat type bhejne par bhi badalna nahi chahiye
        before = self.client.get("/api/machine/M001").get_json()["machine_type"]
        other = "H" if before != "H" else "L"
        self.client.post("/api/predict", json=dict(NORMAL, machine_id="M001", machine_type=other))
        self.assertEqual(self.client.get("/api/machine/M001").get_json()["machine_type"], before)

    # ---------------------------------------------------------- pages
    def test_every_page_loads(self):
        for url in PAGES:
            with self.subTest(url=url):
                r = self.client.get(url)
                self.assertEqual(r.status_code, 200)
                self.assertIn(b"AI PREDICTIVE", r.data)       # sidebar har page par

    def test_sidebar_has_all_sections(self):
        html = self.client.get("/").get_data(as_text=True)
        for label in ("Dashboard", "Machines", "Predict Health", "Analytics", "Anomalies",
                      "Maintenance", "Prediction History", "Model Performance", "Settings"):
            self.assertIn(label, html)

    def test_dashboard_numbers_come_from_database(self):
        s = self.client.get("/api/summary").get_json()
        self.assertEqual(s["total_machines"], s["healthy"] + s["medium"] + s["high"] + s["unknown"])
        self.assertGreaterEqual(s["total_machines"], 50)
        self.assertGreater(s["predictions_today"], 0)
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn(f'<div class="num">{s["total_machines"]}</div>', html)

    def test_demo_data_has_machines_at_risk(self):
        s = self.client.get("/api/summary").get_json()
        self.assertGreater(s["high"] + s["medium"], 0, "demo me koi at-risk machine nahi: dashboard khali dikhega")

    def test_machine_filters(self):
        r = self.client.get("/machines?risk=HIGH")
        self.assertEqual(r.status_code, 200)
        n_high = self.client.get("/api/summary").get_json()["high"]
        # har machine ek table row hai (anomaly badge bhi badge-high class use karta hai, isliye rows gino)
        self.assertEqual(r.get_data(as_text=True).count('<td><a href="/machine/'), n_high)
        self.assertEqual(self.client.get("/machines?q=ZZZ-NOPE").status_code, 200)

    def test_machine_page_shows_status_sections(self):
        high = next(m for m in self.client.get("/api/machines").get_json() if m["risk_level"] == "HIGH")
        html = self.client.get(f"/machine/{high['machine_id']}").get_data(as_text=True)
        for text in ("Failure probability", "Predicted failure", "Maintenance priority",
                     "Why is this machine at risk?", "Sensor trends", "Recent predictions"):
            self.assertIn(text, html)

    def test_figures_route_blocks_path_traversal(self):
        self.assertEqual(self.client.get("/figures/../config.py").status_code, 404)

    def test_invalid_machine_returns_404(self):
        for url in ("/api/machine/NOPE", "/api/history/NOPE", "/machine/NOPE", "/history?machine=NOPE"):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 404)

    # ---------------------------------------------------------- maintenance workflow
    def test_maintenance_form_creates_and_completes_record(self):
        form = dict(machine_id="M001", issue="Heat Dissipation Failure", action="Inspect cooling system",
                    priority="Urgent", scheduled_date="2026-12-01", technician="Technician 1", notes="Check fan")
        r = self.client.post("/maintenance", data=form)
        self.assertEqual(r.status_code, 302)
        orders = self.client.get("/api/maintenance?status=open").get_json()
        mine = [o for o in orders if o["action"] == "Inspect cooling system"]
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["priority"], "Urgent")
        self.assertEqual(mine[0]["status"], "Scheduled")

        done = self.client.post(f"/maintenance/{mine[0]['maintenance_id']}/complete")
        self.assertEqual(done.status_code, 302)
        open_now = self.client.get("/api/maintenance?status=open").get_json()
        self.assertNotIn(mine[0]["maintenance_id"], [o["maintenance_id"] for o in open_now])
        completed = self.client.get("/api/maintenance?status=completed").get_json()
        self.assertIn(mine[0]["maintenance_id"], [o["maintenance_id"] for o in completed])

    def test_maintenance_form_validation(self):
        r = self.client.post("/maintenance", data=dict(machine_id="", action="", priority="Bogus",
                                                        scheduled_date="01-12-2026"))
        self.assertEqual(r.status_code, 400)
        text = r.get_data(as_text=True)
        for msg in ("select a machine", "Action is required", "Priority must be", "valid date"):
            self.assertIn(msg, text)

    def test_maintenance_unknown_machine_rejected(self):
        r = self.client.post("/maintenance", data=dict(machine_id="NOPE", action="x", priority="Low"))
        self.assertEqual(r.status_code, 400)

    def test_complete_unknown_record_is_404(self):
        self.assertEqual(self.client.post("/maintenance/999999/complete").status_code, 404)

    def test_redirect_after_complete_stays_on_this_site(self):
        self.client.post("/maintenance", data=dict(machine_id="M002", action="Redirect check", priority="Low"))
        oid = next(o["maintenance_id"] for o in self.client.get("/api/maintenance").get_json()
                   if o["action"] == "Redirect check")
        r = self.client.post(f"/maintenance/{oid}/complete", data={"next": "https://evil.example/x"})
        self.assertEqual(r.status_code, 302)
        self.assertNotIn("evil.example", r.headers["Location"])
        r2 = self.client.post("/maintenance", data=dict(machine_id="M002", action="Another", priority="Low",
                                                         next="//evil.example"))
        self.assertNotIn("evil.example", r2.headers["Location"])

    def test_maintenance_api(self):
        r = self.client.post("/api/maintenance", json={"machine_id": "M001", "action": "Oil change"})
        self.assertEqual(r.status_code, 201)
        self.assertEqual(self.client.post("/api/maintenance", json={"machine_id": "NOPE", "action": "x"}).status_code, 404)
        self.assertEqual(self.client.post("/api/maintenance", json={"machine_id": "M001"}).status_code, 400)
        bad = self.client.post("/api/maintenance", json={"machine_id": "M001", "action": "x", "priority": "Bogus"})
        self.assertEqual(bad.status_code, 400)

    def test_notes_are_escaped_in_html(self):
        self.client.post("/maintenance", data=dict(machine_id="M003", action="<script>alert(1)</script>",
                                                    notes="<img src=x onerror=alert(2)>", priority="Low"))
        html = self.client.get("/maintenance").get_data(as_text=True)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertNotIn("<img src=x", html)

    # ---------------------------------------------------------- API / health
    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["model_loaded"])

    def test_missing_model_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            app = create_app(db_path=os.path.join(d, "x.db"),
                             model_path=os.path.join(d, "nothing.joblib"), auto_train=False)
            c = app.test_client()
            r = c.get("/api/machines")
            self.assertEqual(r.status_code, 503)
            self.assertIn("Model file not found", r.get_json()["error"])
            page = c.get("/")
            self.assertEqual(page.status_code, 503)
            self.assertIn(b"SYSTEM OFFLINE", page.data)
            self.assertEqual(c.get("/health").status_code, 503)

    def test_corrupt_model_file_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.joblib")
            with open(bad, "wb") as fh:
                fh.write(b"not a model")
            app = create_app(db_path=os.path.join(d, "x.db"), model_path=bad, auto_train=False)
            r = app.test_client().get("/api/machines")
            self.assertEqual(r.status_code, 503)
            self.assertIn("could not be loaded", r.get_json()["error"])

    def test_model_from_other_sklearn_version_is_rejected(self):
        import joblib
        from app import load_artifact
        with tempfile.TemporaryDirectory() as d:
            art = joblib.load(config.MODEL_PATH)
            art["sklearn_version"] = "0.0.1"
            path = os.path.join(d, "old.joblib")
            joblib.dump(art, path)
            with self.assertRaises(RuntimeError):
                load_artifact(path)

    def test_failure_type_model_is_trained_on_failures_only(self):
        classes = list(self.app.config["ARTIFACT"]["failure_type_model"].classes_)
        self.assertNotIn("No Failure", classes)
        self.assertGreaterEqual(len(classes), 2)

    def test_secret_key_is_not_the_old_default(self):
        self.assertNotEqual(self.app.config["SECRET_KEY"], "change-this-in-production")
        self.assertGreaterEqual(len(self.app.config["SECRET_KEY"]), 32)

    def test_predict_page_explains_kelvin(self):
        self.assertIn(b"Kelvin", self.client.get("/predict").data)

    def test_machines_api_lists_seeded_machines(self):
        data = self.client.get("/api/machines").get_json()
        self.assertGreaterEqual(len(data), 50)
        for key in ("risk_level", "failure_probability", "is_anomaly", "health", "open_orders"):
            self.assertIn(key, data[0])

    def test_drift_api(self):
        report = self.client.get("/api/drift").get_json()
        self.assertTrue(report)
        self.assertTrue(all(d["status"] in ("Stable", "Watch", "Drift") for d in report))


class StatusRuleTests(unittest.TestCase):
    """Chhote rule functions (model ki zarurat nahi)."""

    def test_risk_levels_follow_thresholds(self):
        self.assertEqual(risk_from_probability(config.HIGH_THRESHOLD)[0], "HIGH")
        self.assertEqual(risk_from_probability(config.MEDIUM_THRESHOLD)[0], "MEDIUM")
        self.assertEqual(risk_from_probability(config.MEDIUM_THRESHOLD - 0.01)[0], "LOW")

    def test_health_status(self):
        self.assertEqual(health_status("HIGH", False), "CRITICAL")
        self.assertEqual(health_status("MEDIUM", False), "WARNING")
        self.assertEqual(health_status("LOW", True), "WARNING")
        self.assertEqual(health_status("LOW", False), "HEALTHY")

    def test_priority_and_action(self):
        self.assertEqual(priority_for("HIGH"), "Urgent")
        self.assertEqual(priority_for("LOW"), "Low")
        self.assertIn("cooling", recommended_action("HIGH", "Heat Dissipation Failure"))
        self.assertIn("24 hours", recommended_action("HIGH", "Heat Dissipation Failure"))
        self.assertEqual(recommended_action("LOW", "Heat Dissipation Failure"), "Continue monitoring")

    def test_anomaly_level(self):
        self.assertEqual(anomaly_level(0.9, False, 0.5), "LOW")
        self.assertEqual(anomaly_level(0.51, True, 0.5), "MEDIUM")
        self.assertEqual(anomaly_level(0.9, True, 0.5), "HIGH")


class ChartTests(unittest.TestCase):
    def test_empty_and_tiny_data_do_not_crash(self):
        self.assertIn("Not enough data", str(charts.line_chart([1])))
        self.assertIn("Not enough data", str(charts.donut([("a", 0, "red")])))
        self.assertIn("Not enough data", str(charts.hbar([])))
        self.assertIn("Not enough data", str(charts.vbar([("a", 0)])))

    def test_labels_are_escaped(self):
        svg = str(charts.hbar([("<b>x</b>", 3)]))
        self.assertNotIn("<b>x</b>", svg)
        self.assertIn("&lt;b&gt;", svg)

    def test_flat_line_chart_renders(self):
        self.assertIn("<svg", str(charts.line_chart([5, 5, 5, 5], ["a", "b", "c", "d"])))


class PreprocessingTests(unittest.TestCase):
    """Preprocessing functions ke chhote unit tests (model ki zarurat nahi)."""

    def _frame(self):
        ts = pd.date_range("2026-01-01", periods=15, freq="h")
        return pd.DataFrame({
            config.COL_MACHINE: "A1", config.COL_TIME: ts, config.COL_TYPE: "L",
            config.COL_AIR_TEMP: 300.0, config.COL_TEMP: [310 + i * 0.1 for i in range(15)],
            config.COL_SPEED: 1500, config.COL_TORQUE: 40.0, config.COL_WEAR: list(range(15)),
            config.COL_FAILURE: 0,
        })

    def test_rolling_features_use_only_past_data(self):
        full = add_features(self._frame())["temp_roll_mean"].iloc[7]
        short = add_features(self._frame().iloc[:8])["temp_roll_mean"].iloc[7]
        self.assertAlmostEqual(full, short)

    def test_all_model_features_exist(self):
        df = add_features(self._frame())
        for f in config.FEATURES:
            self.assertIn(f, df.columns)
            self.assertFalse(df[f].isna().any(), f"{f} me NaN hai")

    def test_clean_removes_duplicates_and_fills_missing(self):
        df = self._frame()
        df.loc[3, config.COL_TORQUE] = None
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate row
        out = clean(df)
        self.assertEqual(len(out), 15)
        self.assertFalse(out[config.COL_TORQUE].isna().any())

    def test_validate_reading_defaults_air_temperature(self):
        cleaned, errors = validate_reading({"temperature": "310", "rotational_speed": "1500",
                                            "torque": "40", "tool_wear": "10"})
        self.assertEqual(errors, [])
        self.assertAlmostEqual(cleaned[config.COL_AIR_TEMP], 300.0)


if __name__ == "__main__":
    unittest.main()
