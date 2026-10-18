# AI Predictive Maintenance Platform

A machine learning-based web application for monitoring industrial equipment, estimating failure risk, detecting abnormal sensor behaviour, and managing maintenance activities.

The application processes machine sensor readings, generates time-dependent features, runs predictive models, and presents results through a Flask dashboard and REST API. It also supports a maintenance workflow in which a prediction can be converted into a maintenance record and tracked until completion.

**Project type:** B.Tech major project  
**Stack:** Python, Pandas, scikit-learn, XGBoost, Flask, SQLite

## Table of Contents

- [Features](#features)
- [Application Pages](#application-pages)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Dataset](#dataset)
- [Processing and Prediction Workflow](#processing-and-prediction-workflow)
- [Risk Classification](#risk-classification)
- [Model Evaluation](#model-evaluation)
- [REST API](#rest-api)
- [Tests](#tests)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Potential Improvements](#potential-improvements)
- [Project Report](#project-report)

## Features

| Component | Description |
|---|---|
| Data pipeline | Loads CSV data, validates values, handles missing values and duplicates, and sorts readings by timestamp. |
| Feature engineering | Creates rolling statistics, deltas, trend slopes, power-related features, torque-to-speed ratios, and tool-wear features. |
| Failure prediction | Uses a gradient-boosting classifier to estimate failure probability and assign a LOW, MEDIUM, or HIGH risk level. XGBoost is used when available; the application can fall back to scikit-learn's `HistGradientBoostingClassifier`. |
| Baseline comparison | Evaluates Logistic Regression, Decision Tree, Random Forest, and the primary gradient-boosting model. |
| Anomaly detection | Uses Isolation Forest to flag unusual sensor behaviour and assign an anomaly level. |
| Failure-type classification | Uses a Random Forest multi-class classifier to estimate a likely failure category. |
| Remaining Useful Life (RUL) | Includes an experimental Random Forest regressor that estimates the number of readings remaining before failure. |
| Risk explanation | Estimates the contribution of sensor groups to a prediction using a model-based what-if method. This is not a SHAP implementation. |
| Dashboard | Provides nine application pages for equipment status, predictions, analytics, anomalies, maintenance, history, model performance, and settings. |
| Maintenance workflow | Converts an at-risk machine into a maintenance record, assigns priority, technician, and scheduled date, and tracks completion. |
| REST API | Exposes endpoints for predictions, machine data, prediction history, maintenance records, drift information, summaries, and health checks. |
| Persistence | Stores machines, readings, failures, predictions, maintenance records, and model-version information in SQLite. |
| Monitoring | Displays prediction history, high-risk machines, and a basic data-drift indicator. |
| Automated tests | Includes 46 tests covering validation, risk rules, pages, maintenance workflow, API behaviour, charts, and leakage checks. |

## Application Pages

| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Summary cards, risk trends, health distribution, machines requiring attention, and recent alerts. Values are sourced from the database. |
| Machines | `/machines` | Lists machines with health, risk, failure probability, anomaly status, and RUL. Supports risk, anomaly, and search filters. |
| Machine details | `/machine/<id>` | Shows current status, predicted failure, priority, risk explanation, RUL, sensor trends, probability history, maintenance records, and previous failures. |
| Predict Health | `/predict` | Accepts a machine selection or sensor inputs and displays the prediction, probability, likely failure type, recommended action, and explanation. |
| Analytics | `/analytics` | Displays failure distribution, risk distribution, health trends, and expected failure types. |
| Anomalies | `/anomalies` | Lists machines with unusual behaviour and their associated risk. |
| Maintenance | `/maintenance` | Supports maintenance record creation, open/completed record tracking, and completion updates. |
| Prediction History | `/history` | Displays saved predictions by machine and a probability history chart. |
| Model Performance | `/model` | Shows model comparison metrics, selected-model scores, confusion matrix, and plots. |
| Settings | `/settings` | Displays model information, thresholds, database counts, and a read-only drift check. |

Charts are rendered as inline SVG using `src/charts.py`; the application does not require a JavaScript charting library or an internet connection to render them.

## Architecture

```text
Sensor Readings / CSV
          |
          v
Data Cleaning and Validation
          |
          v
Feature Engineering
          |
          +----------------------+--------------------+
          |                      |                    |
          v                      v                    v
 Failure Prediction       Anomaly Detection       RUL Estimation
          |                      |                    |
          +----------------------+--------------------+
                                 |
                                 v
                  Risk, Failure Type and Explanation
                                 |
                                 v
                         Flask Application
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
              Web Dashboard               REST API
                    |                         |
                    +------------+------------+
                                 |
                                 v
                           SQLite Database
                                 |
                                 v
                     Maintenance Workflow
```

## Project Structure

```text
predictive-maintenance/
├── app.py                     # Flask application, web pages, and API routes
├── config.py                  # Paths, thresholds, actions, and feature configuration
├── requirements.txt           # Python dependencies
├── data/
│   ├── raw/
│   │   └── predictive_maintenance.csv
│   ├── processed/
│   │   └── features.csv        # Generated during training
│   ├── external/
│   │   └── ai4i2020.csv        # Optional real AI4I dataset
│   └── maintenance.db         # SQLite database, created when needed
├── models/
│   └── predictive_maintenance.joblib
├── notebooks/
│   ├── 01_eda
│   ├── 02_feature_engineering
│   ├── 03_model_training
│   └── 04_real_ai4i_comparison
├── reports/
│   ├── metrics.json
│   ├── model_comparison.csv
│   └── figures/
├── src/
│   ├── data_generator.py      # Synthetic dataset generator
│   ├── import_ai4i.py         # Converts AI4I CSV to the project format
│   ├── preprocessing.py       # Cleaning, feature engineering, validation
│   ├── model_utils.py         # Model loading, prediction, and explanations
│   ├── status.py              # Risk, health, priority, and action rules
│   ├── charts.py              # SVG chart generation
│   ├── train.py               # Training pipeline
│   └── database.py            # SQLite operations
├── templates/                 # Flask HTML templates
├── static/
│   └── css/
│       └── style.css
├── tests/
│   └── test_app.py
└── docs/
    ├── VIVA_NOTES.md
    └── report/
        ├── Project_Report_Gaurav_Kumar.docx
        └── Project_Report_Gaurav_Kumar.pdf
```

Generated files may not exist until the training or application setup process has run. The exact files present depend on the current project checkout and the datasets supplied.

## Requirements

- Python 3.10 or later
- `pip`
- The dependencies listed in `requirements.txt`

A virtual environment is recommended to keep project dependencies isolated.

## Installation

### Windows (Command Prompt or PowerShell)

```bat
cd AI_Predictive_Maintenance
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux or macOS

```bash
cd AI_Predictive_Maintenance
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Conda

```bash
conda create -n predictive-maintenance python=3.11 -y
conda activate predictive-maintenance
pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Open the following address in a browser:

```text
http://127.0.0.1:5000/
```

On the first run, the application is configured to prepare the model and database if required. Depending on the files already present, this may include:

1. Training a model if the model file is missing.
2. Creating and populating the SQLite database.
3. Saving predictions for recent machine readings so that the dashboard and history pages have data to display.

For a step-by-step run, use:

```bash
# Optional: generate the synthetic dataset
python -m src.data_generator

# Train models and generate reports/plots
python -m src.train

# Start the web application
python app.py

# Run automated tests
python -m unittest discover -s tests -v
```

To work with the notebooks, install Jupyter and the IPython kernel:

```bash
pip install notebook ipykernel
jupyter notebook
```

Then open the `notebooks/` directory.

## Dataset

### Default dataset

The default file is:

```text
data/raw/predictive_maintenance.csv
```

It contains **10,000 rows**, representing **50 machines with 200 hourly readings each**. The reported failure rate is **2.96%**.

**Important:** The default dataset is synthetic and is generated by `src/data_generator.py`. Its columns and failure rules are designed to resemble the AI4I 2020 Predictive Maintenance dataset, but it is not actual factory data. Performance on this dataset should not be treated as a guarantee of performance in a production plant.

Some generated machines follow a degradation pattern near the end of their timelines: tool wear increases, rotational speed decreases, torque increases, and cooling behaviour deteriorates. This is why some machines can appear as HIGH risk in the demo. The time-based test set can also have a different failure rate from the training set because degradation is concentrated later in the timeline.

### Dataset columns

| Column | Description |
|---|---|
| `machine_id` | Equipment identifier |
| `timestamp` | Time of the reading |
| `machine_type` | Machine quality variant (`L`, `M`, or `H`) |
| `air_temperature` | Air temperature in Kelvin |
| `temperature` | Process temperature in Kelvin |
| `rotational_speed` | Rotational speed in RPM |
| `torque` | Torque in Nm |
| `tool_wear` | Tool-wear value in minutes |
| `failure` | Target label (`0` = no failure, `1` = failure) |
| `failure_type` | Failure category |

### Demo timestamps

The configuration setting `SHIFT_SEED_TO_NOW = True` in `config.py` shifts seed timestamps so that the latest reading appears to be current. This supports demo elements such as the “Predictions today” card.

For real historical data, set `SHIFT_SEED_TO_NOW = False` and verify the database contents before evaluating date-dependent dashboard metrics.

### Using the AI4I 2020 dataset

1. Obtain the AI4I 2020 Predictive Maintenance CSV from its original source or a trusted dataset repository.
2. Convert it to the project format:

   ```bash
   python -m src.import_ai4i path/to/ai4i2020.csv
   ```

3. Follow the project's data-reset procedure before retraining: back up any database you need, remove `data/maintenance.db` if it is safe to recreate, set `SHIFT_SEED_TO_NOW = False`, and run:

   ```bash
   python -m src.train
   ```

The AI4I dataset does not provide machine IDs or timestamps in the format expected by this project. The converter assigns hourly timestamps in row order and treats each group of 200 rows as a machine. Consequently, rolling and trend features may be less representative of real machine histories when applied to this dataset.

## Processing and Prediction Workflow

1. **Cleaning and validation** — `preprocessing.clean` handles duplicate records, converts impossible values to missing values, applies machine-wise forward filling, and sorts readings by timestamp.
2. **Feature engineering** — `preprocessing.add_features` calculates rolling and difference-based features using previous readings. The same feature-generation logic is used during training and prediction.
3. **Time-based split** — The first 75% of the timeline is used for training and the last 25% for testing. This is intended to reflect prediction on later observations rather than a random split.
4. **Class imbalance handling** — Failures are relatively rare. Class weighting is used, and evaluation includes precision, recall, F1, ROC-AUC, and PR-AUC rather than accuracy alone.
5. **Prediction** — A request from the web form or API is combined with the machine's previous readings from the database. The application generates features, runs the relevant models, derives risk and status information, produces an explanation, and saves the prediction.

### Risk classification

The default thresholds are configured in `config.py`.

| Failure probability | Risk level | Health status | Priority | Recommended response |
|---|---|---|---|---|
| `>= 0.80` | HIGH | CRITICAL | Urgent | Inspect according to the predicted failure type; target action within 24 hours. |
| `>= 0.50` and `< 0.80` | MEDIUM | WARNING | Medium | Inspect according to the predicted failure type; target action within 7 days. |
| `< 0.50` | LOW | HEALTHY, or WARNING if an anomaly is present | Low | Continue monitoring. |

These are application rules, not universal industrial safety thresholds. Thresholds, time frames, and failure-type actions can be changed in `config.py`.

### Risk explanation

The application groups 17 features into eight sensor-related groups. It replaces one group at a time with its training-median values and observes how the model's failure score changes. A larger decrease suggests that the group contributed more to the original score.

This is a lightweight, model-based what-if explanation, not SHAP. It is generated for MEDIUM and HIGH predictions.

## Model Evaluation

### Synthetic dataset results

The following results are from a time-based test run with **2,500 rows and 133 actual failures**.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.17 | 0.97 | 0.29 | 0.937 | 0.491 |
| Decision Tree | 0.65 | 0.92 | 0.76 | 0.934 | 0.879 |
| Random Forest | 0.97 | 0.77 | 0.86 | 0.977 | 0.933 |
| Gradient boosting (primary) | 0.91 | 0.89 | 0.90 | 0.978 | 0.937 |

For the primary model, the reported confusion matrix counts are:

- True negatives: 2,355
- False positives: 12
- False negatives: 15
- True positives: 118

These figures correspond to a run in which scikit-learn's `HistGradientBoostingClassifier` was used as the fallback instead of XGBoost. Installing XGBoost and retraining may produce different results. The metrics for the current run are saved in `reports/metrics.json` and shown on the Model Performance page.

### Other model results

- **Failure-type classifier:** Trained and tested on failure rows only (163 training rows and 133 test rows in the reported run). Weighted F1 was 0.95 and macro F1 was 0.79. Most failures belong to the Heat Dissipation category, while other categories have very few test examples (2–13). Scores for those smaller categories should be interpreted cautiously.
- **Anomaly detector:** Reported precision and recall against actual failures were approximately 48% and 47%. Treat anomaly detection as a supporting signal, not a replacement for the failure classifier.
- **RUL regressor:** Experimental MAE was 7.7 readings, compared with 10.2 for a mean-prediction baseline. This is an improvement on that test, but the output remains a rough estimate and depends on the quality of run-to-failure data.

### Real AI4I 2020 experiment

The project notes report an F1 score of **0.79** and ROC-AUC of **0.984** for the primary model on the AI4I 2020 dataset. Outputs are stored under `reports/ai4i_real/`, with analysis in notebook `04_real_ai4i_comparison`.

These figures are specific to the reported experiment. Because the AI4I rows are not a genuine time series with machine IDs and timestamps, rolling features have limited temporal meaning. Do not compare these results directly with the synthetic-data results without accounting for the different data construction and evaluation setup.

## REST API

The Flask application exposes the following endpoints.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Runs a prediction and returns results including explanation, action, and priority. |
| `GET` | `/api/summary` | Returns dashboard summary values. |
| `GET` | `/api/machines` | Returns machine information, latest risk/anomaly information, and open maintenance orders. |
| `GET` | `/api/machine/<id>` | Returns machine details and its latest prediction. |
| `GET` | `/api/history/<id>?limit=50` | Returns prediction history for a machine. |
| `GET` | `/api/maintenance?status=open` | Returns maintenance records; `status` can be `open`, `completed`, or omitted for all records. |
| `POST` | `/api/maintenance` | Creates a maintenance record. |
| `GET` | `/api/drift` | Returns the data-drift report. |
| `GET` | `/health` | Provides an application health check. |

### Prediction request example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M001",
    "air_temperature": 300,
    "temperature": 308,
    "rotational_speed": 1250,
    "torque": 55,
    "tool_wear": 230
  }'
```

Required fields:

- `machine_id`
- `temperature`
- `rotational_speed`
- `torque`
- `tool_wear`

Optional fields:

- `air_temperature` — if omitted, the application derives it as `temperature - 10`.
- `machine_type` — used for a new machine; defaults to `L`.

### Maintenance request fields

A maintenance request requires `machine_id` and `action`. Optional fields are:

- `issue`
- `priority` (`Urgent`, `High`, `Medium`, or `Low`)
- `scheduled_date` in `YYYY-MM-DD` format
- `technician`
- `notes`

## Tests

Run the automated test suite with:

```bash
python -m unittest discover -s tests -v
```

The project currently documents 46 automated tests covering input validation, risk rules, application pages, maintenance workflow, API behaviour, chart output, and checks related to data leakage.

## Known Limitations

- The default dataset is synthetic; performance on real industrial equipment may differ significantly.
- The application does not currently implement login or authentication. It is intended as a single-user demonstration, and CSRF protection is not configured for forms. Add appropriate authentication, authorization, and CSRF protection before exposing it to users.
- The application uses Flask's development server by default. Use a production WSGI server such as Gunicorn or Waitress for deployment.
- Anomaly detection is a supporting signal, and RUL is experimental.
- Live sensor ingestion through MQTT or an IoT pipeline is not implemented. Readings are submitted through the web form or API.
- The AI4I conversion process synthesizes timestamps and machine groupings; temporal features therefore require careful interpretation.

## Troubleshooting

| Problem | Suggested action |
|---|---|
| Model file is missing | Run `python -m src.train`. |
| The saved model cannot be loaded | The model may have been saved with a different scikit-learn version. Recreate it using `python -m src.train` in the active environment. |
| Dashboard shows old data after changing the dataset | Back up any required data, remove `data/maintenance.db` if it can safely be recreated, and restart the application. |
| Installing XGBoost fails | The application can use its fallback gradient-boosting classifier, if the installed project dependencies support that path. |
| Port 5000 is already in use | Change the port in the application's run configuration in `app.py`. |

## Potential Improvements

- Add live sensor ingestion through MQTT or a dedicated IoT/REST pipeline.
- Evaluate RUL models using real run-to-failure sequence data, such as NASA C-MAPSS.
- Add SHAP-based explanations.
- Introduce PostgreSQL and a user authentication system.
- Add email or webhook alerts for high-risk predictions.
- Add model experiment and version tracking with MLflow.
- Deploy the application to a cloud platform such as Azure or Render, with production configuration and security controls.

## Project Report

The project report is available at:

```text
docs/report/Project_Report_Gaurav_Kumar.docx
docs/report/Project_Report_Gaurav_Kumar.pdf
```

Update highlighted placeholders in the certificate and declaration sections (such as guide name, institution, roll number, and date) before submitting the report.
