# AI Predictive Maintenance Platform

Machine learning based system jo industrial equipment ke sensor readings dekhkar batata hai ki machine
fail hone wali hai ya nahi, kaun sa failure aa sakta hai, kyun, aur maintenance team ko kya karna chahiye.
Sirf prediction nahi: prediction se maintenance record banane aur use complete karne tak ka poora workflow.

B.Tech major project - Python, Pandas, Scikit-Learn, XGBoost, Flask, SQLite.

---

## Features

| Module | Kya karta hai |
|---|---|
| Data pipeline | CSV load, cleaning, validation, timestamp sorting, duplicate/missing handling |
| Feature engineering | Rolling mean/std, delta, trend slope, power, torque/speed ratio, wear rate |
| Failure prediction | XGBoost classifier (failure probability + LOW / MEDIUM / HIGH risk) |
| Baseline comparison | Logistic Regression, Decision Tree, Random Forest vs primary model |
| Anomaly detection | Isolation Forest, LOW / MEDIUM / HIGH anomaly level |
| Failure type | Random Forest multi-class (Heat Dissipation, Overstrain, Tool Wear, Power, Random) |
| RUL (experimental) | Random Forest regressor, agle failure me kitni readings bachi hain |
| Explainability | "Why is this machine at risk?" - sensor group ke hisaab se risk ka share (%) |
| Dashboard UI | Sidebar wala industrial dashboard, 9 pages (neeche list hai) |
| Maintenance workflow | Prediction se maintenance record banao, priority/technician/date do, complete mark karo |
| REST API | Predict, summary, machines, history, maintenance, drift, health |
| Database | SQLite - machines, readings, failures, predictions, maintenance, model versions |
| Monitoring | Prediction history, high-risk alerts, simple data drift indicator |
| Tests | 46 automated tests (validation, risk rules, pages, maintenance flow, API, charts, leakage check) |

---

## UI pages

| Page | URL | Kya dikhata hai |
|---|---|---|
| Dashboard | `/` | 6 cards (total, healthy, medium, high, predictions today, maintenance due), risk trend, health donut, machines needing attention, recent alerts. Saare numbers database se aate hain |
| Machines | `/machines` | Saari machines: health, risk, probability, anomaly, RUL. Risk / anomaly / search filter |
| Machine details | `/machine/<id>` | Status, failure probability, predicted failure, priority, "why at risk" bars, RUL, 4 sensor trend cards, probability history chart, maintenance records, past failures |
| Predict Health | `/predict` | Machine dropdown (latest reading se auto-fill) + sensor inputs. Result page: probability, risk, likely failure, recommended action, explanation |
| Analytics | `/analytics` | Failure distribution, risk distribution, health trend, expected failure types |
| Anomalies | `/anomalies` | Abnormal behaviour wali machines aur unka risk |
| Maintenance | `/maintenance` | At-risk machines, record banane ka form, open/completed records, "Mark completed" |
| Prediction History | `/history` | Machine-wise saari predictions + probability chart |
| Model Performance | `/model` | Model comparison table, selected model ke scores, confusion matrix, plots |
| Settings | `/settings` | Model info, thresholds, database counts, drift check (read-only) |

Charts inline SVG hain (`src/charts.py`), isliye koi JavaScript library ya internet connection nahi chahiye.

---

## Folder structure

```
predictive-maintenance/
|-- app.py                  Flask application (pages, API)
|-- config.py               Saari settings: paths, thresholds, actions, feature list
|-- requirements.txt
|-- data/
|   |-- raw/                predictive_maintenance.csv (dataset)
|   |-- processed/          features.csv (training ke baad)
|   `-- maintenance.db      SQLite database (pehli run par ban jaata hai)
|-- models/                 predictive_maintenance.joblib (training ke baad)
|-- notebooks/              01_eda, 02_feature_engineering, 03_model_training, 04_real_ai4i_comparison (outputs ke saath)
|-- reports/                metrics.json, model_comparison.csv, figures/ (plots)
|-- src/
|   |-- data_generator.py   Dataset generator
|   |-- import_ai4i.py      Real AI4I 2020 CSV ko is format me badalta hai
|   |-- preprocessing.py    Cleaning + feature engineering + input validation
|   |-- model_utils.py      Model banana, explanation, predict
|   |-- status.py           Risk / health / priority / recommended action rules
|   |-- charts.py           SVG charts
|   |-- train.py            Poora training pipeline
|   `-- database.py         SQLite functions
|-- templates/              HTML pages (base, dashboard, machines, machine, predict, result,
|                           analytics, anomalies, maintenance, history, model_performance,
|                           settings, error, _macros)
|-- static/css/style.css
|-- tests/test_app.py
`-- docs/VIVA_NOTES.md      Viva / interview ke liye notes
```

---

## Setup

Python 3.10 ya usse naya chahiye.

**Windows**
```
cd AI_Predictive_Maintenance
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS**
```
cd AI_Predictive_Maintenance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Conda**
```
conda create -n AI_Predictive_Maintenance python=3.11 -y
conda activate predictive-maintenance
pip install -r requirements.txt
```

## Run

```
python app.py
```

Browser me kholo: http://127.0.0.1:5000/

Pehli baar run karne par app khud ye kaam karti hai (1-2 minute):
1. Model file na mile to training chalati hai (plots aur metrics bhi ban jaate hain)
2. Database banati hai aur dataset se bharti hai
3. Har machine ki recent readings par predictions save karti hai, taaki dashboard aur history khali na dikhein

Step by step chalana ho:

```
python -m src.data_generator     # dataset banao (data/raw/ me pehle se hai, optional)
python -m src.train              # training, plots, model save
python app.py                    # web app
python -m unittest discover -s tests -v    # tests
```

Notebooks ke liye: `pip install notebook ipykernel`, phir `jupyter notebook` aur `notebooks/` folder kholo.

---

## Dataset

`data/raw/predictive_maintenance.csv` - 50 machines x 200 hourly readings = 10,000 rows, 2.96% failures.

**Dhyan dein:** ye dataset `src/data_generator.py` se generate kiya gaya hai (synthetic). Columns aur failure rules
AI4I 2020 dataset jaise rakhe gaye hain. Isliye model ke scores asli factory data par same hone ki guarantee nahi
hai. Report/viva me ise clearly bolna chahiye.

Generator me kuch machines "degrading" hoti hain: timeline ke end me unka wear badhta hai, speed girti hai, torque
badhta hai aur cooling kharab hoti hai. Isi wajah se demo me kuch machines abhi HIGH risk dikhti hain. Iska ek
side effect: degradation timeline ke aakhri hisse me hoti hai, isliye time-based test period me failure rate
(5.3%) train period (2.2%) se zyada hai.

**Demo data ki timestamps:** `config.py` me `SHIFT_SEED_TO_NOW = True` hai, jisse database bharte waqt saari
timestamps aise aage khisak jaati hain ki sabse nayi reading "abhi" ki ho (taaki "Predictions today" card live
dikhe). Real data use karo to ise `False` kar do.

Real dataset use karna ho:
1. AI4I 2020 Predictive Maintenance CSV download karo (Kaggle / UCI)
2. `python -m src.import_ai4i path/to/ai4i2020.csv`
3. `data/maintenance.db` delete karo, `SHIFT_SEED_TO_NOW = False` karo, phir `python -m src.train` chalao

AI4I me machine ID aur timestamp nahi hota, isliye converter rows ko order me hourly timestamp deta hai aur
har 200 rows ko ek machine maan leta hai. Rolling/trend features is data par kam meaningful honge.

| Column | Meaning |
|---|---|
| machine_id | Equipment ID |
| timestamp | Reading time |
| machine_type | L / M / H (quality variant) |
| air_temperature | Kelvin |
| temperature | Process temperature, Kelvin |
| rotational_speed | rpm |
| torque | Nm |
| tool_wear | minutes |
| failure | 0 / 1 (target) |
| failure_type | Failure category |

---

## Kaise kaam karta hai

```
Sensors / CSV -> Cleaning -> Features -> Boosted model + Isolation Forest + RUL -> Flask -> Dashboard
```

1. **Cleaning** (`preprocessing.clean`): duplicates hatana, impossible values NaN banana, machine-wise forward fill,
   timestamp ke hisaab se sort.
2. **Features** (`preprocessing.add_features`): saare rolling aur diff features sirf pichli readings se bante hain,
   isliye future ka data leak nahi hota. Yahi function training aur Flask dono me chalta hai.
3. **Split**: random nahi, time-based. Pehla 75% time train, last 25% test.
4. **Imbalance**: failures kam hain, isliye class weight use kiya aur accuracy ke bajay precision, recall, F1,
   ROC-AUC, PR-AUC dekhe.
5. **Prediction**: web form ya API se reading aati hai -> us machine ki pichli readings database se -> features ->
   models -> risk, anomaly, failure type, RUL, explanation -> database me save.

### Risk levels (`config.py`)

| Failure probability | Risk | Health | Priority | Recommended action |
|---|---|---|---|---|
| >= 0.80 | HIGH | CRITICAL | Urgent | Failure type ke hisaab se check, "within 24 hours" |
| >= 0.50 | MEDIUM | WARNING | Medium | Failure type ke hisaab se check, "within 7 days" |
| < 0.50 | LOW | HEALTHY (WARNING agar anomaly ho) | Low | Continue monitoring |

Thresholds, time-frame aur har failure type ka action `config.py` me badal sakte ho.

### "Why is this machine at risk?"

17 features ko 8 sensor groups me baanta hai (Tool wear, Process - air temperature, Torque, Rotational speed,
...). Har group ko ek ek karke training median se replace karke dekhte hain ki model ka failure score kitna girta
hai. Jo group hatane par sabse zyada girta hai wahi risk badha raha tha. Bars us score ka hissa (%) dikhate hain.
Ye SHAP nahi hai, SHAP ka simple "what-if" version hai (bina extra library ke). Explanation sirf MEDIUM/HIGH
predictions ke liye banta hai.

---

## Results (is dataset par, time-based test split)

Test set: 2,500 rows, 133 real failures.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.17 | 0.97 | 0.29 | 0.937 | 0.491 |
| Decision Tree | 0.65 | 0.92 | 0.76 | 0.934 | 0.879 |
| Random Forest | 0.97 | 0.77 | 0.86 | 0.977 | 0.933 |
| Gradient boosting (primary) | 0.91 | 0.89 | 0.90 | 0.978 | 0.937 |

Primary model confusion matrix: 2355 healthy sahi, 12 false alarms, 15 missed failures, 118 failures pakde.

> Ye numbers us run ke hain jisme XGBoost ki jagah sklearn ka `HistGradientBoostingClassifier` use hua (app XGBoost
> na milne par automatically fallback leti hai). `pip install xgboost` ke baad training chalane par XGBoost use hoga
> aur numbers thode alag aa sakte hain. Training ke baad `reports/metrics.json` aur Model Performance page par
> aapke run ke asli numbers milenge.

Doosre modules:
- **Failure type classifier**: sirf failure rows par train/test (163 / 133). Weighted F1 0.95, macro F1 0.79. Zyadatar failures Heat Dissipation type ke hain aur baaki types ke test examples bahut kam hain (2-13), isliye unke scores par zyada bharosa nahi karna chahiye.
- **Anomaly detector**: real failures ke saath ~48% precision / 47% recall. Ise failure model ke saath
  supporting signal ki tarah use karna chahiye, uski jagah nahi.
- **RUL**: experimental. MAE 7.7 readings, jabki sirf average predict karne par MAE 10.2 aata hai, yaani behtar hai
  par rough indicator hi hai. Real run-to-failure data (NASA C-MAPSS) par ye module zyada meaningful hoga.

---

## API

| Method | Endpoint | Kaam |
|---|---|---|
| POST | `/api/predict` | JSON prediction (result me explanation, action, priority bhi) |
| GET | `/api/summary` | Dashboard cards ke numbers |
| GET | `/api/machines` | Saari machines + latest risk, anomaly, open orders |
| GET | `/api/machine/<id>` | Machine details + latest prediction |
| GET | `/api/history/<id>?limit=50` | Prediction history |
| GET | `/api/maintenance?status=open` | Maintenance records (`open` / `completed` / sab) |
| POST | `/api/maintenance` | Maintenance record add |
| GET | `/api/drift` | Data drift report |
| GET | `/health` | Health check |

Example:

```
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"M001","air_temperature":300,"temperature":308,
       "rotational_speed":1250,"torque":55,"tool_wear":230}'
```

Required fields: `machine_id`, `temperature`, `rotational_speed`, `torque`, `tool_wear`.
Optional: `air_temperature` (na dene par `temperature - 10`), `machine_type` (sirf nayi machine ke liye, default L).

Maintenance record: `machine_id` aur `action` required; `issue`, `priority` (Urgent / High / Medium / Low),
`scheduled_date` (YYYY-MM-DD), `technician`, `notes` optional.

---

## Limitations

- Data synthetic hai, real plant par accuracy alag hogi.
- Login / authentication nahi hai (single-user demo), isliye forms par CSRF protection bhi nahi hai. Internet par
  deploy karne se pehle ye add karna chahiye.
- Flask ka development server hai; production me gunicorn / waitress ke peeche chalao.
- Anomaly detector aur RUL supporting / experimental hain.
- Live sensor streaming (MQTT / IoT) nahi hai; readings form ya API se aati hain.

## Common problems

| Problem | Solution |
|---|---|
| `Model file not found` error page | `python -m src.train` chalao |
| `The model could not be loaded` | Alag scikit-learn version se save hui file hai, `python -m src.train` dobara chalao |
| Naye dataset ke baad dashboard purana data dikhaye | `data/maintenance.db` delete karke app dobara chalao |
| `pip install xgboost` fail | App fallback model se bhi chalti hai, XGBoost optional hai |
| Port 5000 busy | `app.py` ke last line me `port=5000` badal do |

---

## Future improvements

- Live IoT data (MQTT / REST se sensor readings)
- LSTM based RUL on real sequence data (NASA C-MAPSS)
- SHAP explanations
- PostgreSQL + login system (`user_session` table schema me pehle se hai)
- Email / webhook alerts
- MLflow se model version tracking
- Cloud deployment (Render / Azure)


## Real AI4I 2020 par result
`data/external/ai4i2020.csv` me asli AI4I 2020 dataset rakha hai. `python -m src.import_ai4i data/external/ai4i2020.csv`
se project format me badal kar train karne par primary model ka F1 0.79 aur ROC-AUC 0.984 aaya
(`reports/ai4i_real/`, notebook `04_real_ai4i_comparison`). Synthetic data se kam hai kyunki AI4I ki rows time series
nahi hain, isliye rolling features kam kaam aate hain. Is experiment ke baad `data/maintenance.db` delete karke aur
`config.SHIFT_SEED_TO_NOW = False` karke hi app chalana.

## Project report
`docs/report/Project_Report_Gaurav_Kumar.docx` (aur `.pdf`) me poori report hai. Certificate / declaration me
highlighted jagah (guide ka naam, college, roll number, date) apni details se bhar lena.

## Model file ke baare me
`models/predictive_maintenance.joblib` ek scikit-learn version me bani hai. Alag version install ho to app
khud dobara train kar leti hai (kuch second lagte hain). Naye training run me numbers thode alag aa sakte hain.
