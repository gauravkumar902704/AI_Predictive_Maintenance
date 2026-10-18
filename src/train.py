"""
train.py
--------
Poora training pipeline:
    1. data load + clean + feature engineering
    2. EDA plots (reports/figures/)
    3. time-based train/test split (leakage se bachne ke liye)
    4. baseline models vs primary boosted model ka comparison
    5. anomaly detector (Isolation Forest), failure-type classifier, RUL regressor
    6. sab kuch ek joblib artifact me save

Run:
    python -m src.train
"""
import json
import os
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # bina screen ke bhi plots save ho jaayein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score,
                             precision_recall_curve, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.model_utils import make_boosted_classifier  # noqa: E402
from src.preprocessing import build_training_frame  # noqa: E402

sns.set_theme(style="whitegrid")


def save_fig(name):
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    path = os.path.join(config.FIGURES_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved plot: {name}")


# --------------------------------------------------------------------- EDA
def run_eda(df):
    """Har plot ek sawaal ka jawab deta hai (README me explain kiya hai)."""
    print("\n[EDA] plots bana raha hoon...")

    # 1. class imbalance
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x=config.COL_FAILURE)
    ax.set_title("Class Balance (0 = healthy, 1 = failure)")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    save_fig("01_class_balance.png")

    # 2. failure types
    plt.figure(figsize=(7, 4))
    counts = df.loc[df[config.COL_FAILURE] == 1, config.COL_FAILURE_TYPE].value_counts()
    sns.barplot(x=counts.values, y=counts.index)
    plt.title("Failure Types")
    plt.xlabel("count")
    save_fig("02_failure_types.png")

    # 3. healthy vs failed distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.ravel(), [config.COL_TEMP, config.COL_SPEED,
                                      config.COL_TORQUE, config.COL_WEAR]):
        sns.histplot(data=df, x=col, hue=config.COL_FAILURE, kde=True,
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(f"{col} by failure status")
    save_fig("03_distributions_by_failure.png")

    # 4. correlation heatmap
    cols = [config.COL_AIR_TEMP, config.COL_TEMP, config.COL_SPEED, config.COL_TORQUE,
            config.COL_WEAR, "power", "temp_diff", config.COL_FAILURE]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Sensor Correlation Heatmap")
    save_fig("04_correlation_heatmap.png")

    # 5. box plot of torque by failure
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=config.COL_FAILURE, y=config.COL_TORQUE)
    plt.title("Torque: healthy vs failure")
    save_fig("05_boxplot_torque.png")

    # 6. machine-wise failure count (top 10)
    plt.figure(figsize=(8, 4))
    mcount = df.groupby(config.COL_MACHINE)[config.COL_FAILURE].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=mcount.index, y=mcount.values)
    plt.title("Top 10 machines by failure count")
    plt.xticks(rotation=45)
    save_fig("06_machine_failures.png")

    # 7. ek machine ka sensor trend failure points ke saath
    worst = mcount.index[0]
    m = df[df[config.COL_MACHINE] == worst]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, col in zip(axes, [config.COL_TEMP, config.COL_TORQUE, config.COL_WEAR]):
        ax.plot(m[config.COL_TIME], m[col], linewidth=1)
        f = m[m[config.COL_FAILURE] == 1]
        ax.scatter(f[config.COL_TIME], f[col], color="red", s=25, zorder=3, label="failure")
        ax.set_ylabel(col)
    axes[0].set_title(f"Sensor trend - {worst} (red = failure)")
    axes[0].legend()
    save_fig("07_sensor_trend.png")


# --------------------------------------------------------------------- helpers
def evaluate(name, y_true, prob, threshold=0.5):
    """Ek model ke saare classification metrics nikalta hai."""
    pred = (prob >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
    }


def time_based_split(df):
    """
    Random split nahi karte kyunki future ka data train me aa jaayega.
    Time ke hisaab se pehle 75% train, last 25% test.
    """
    cutoff = df[config.COL_TIME].quantile(1 - config.TEST_FRACTION)
    train = df[df[config.COL_TIME] <= cutoff]
    test = df[df[config.COL_TIME] > cutoff]
    return train.copy(), test.copy(), cutoff


# --------------------------------------------------------------------- main
def main():
    start = time.time()
    print("=" * 60)
    print("AI Predictive Maintenance - training shuru")
    print("=" * 60)

    # ---- 1. data
    df = build_training_frame()
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Rows: {len(df)} | Machines: {df[config.COL_MACHINE].nunique()} "
          f"| Failure rate: {df[config.COL_FAILURE].mean() * 100:.2f}%")

    # ---- 2. EDA
    run_eda(df)

    # ---- 3. split
    train, test, cutoff = time_based_split(df)
    features = config.FEATURES
    X_train, y_train = train[features], train[config.COL_FAILURE].values
    X_test, y_test = test[features], test[config.COL_FAILURE].values
    print(f"\nSplit cutoff time: {cutoff}")
    print(f"Train: {len(train)} rows ({y_train.sum()} failures) | "
          f"Test: {len(test)} rows ({y_test.sum()} failures)")

    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    # ---- 4. model comparison
    print("\n[MODELS] baseline vs primary model")
    boosted, algo_name = make_boosted_classifier(pos_weight)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE),
        algo_name: boosted,
    }

    results, probs = [], {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        p = mdl.predict_proba(X_test)[:, 1]
        probs[name] = p
        r = evaluate(name, y_test, p)
        results.append(r)
        print(f"  {name:<40} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f}")

    comparison = pd.DataFrame(results)

    # primary model
    primary_name = algo_name
    primary = models[primary_name]
    prob_primary = probs[primary_name]
    pred_primary = (prob_primary >= 0.5).astype(int)

    print(f"\nPrimary model: {primary_name}")
    print(classification_report(y_test, pred_primary, zero_division=0,
                                target_names=["healthy", "failure"]))
    cm = confusion_matrix(y_test, pred_primary)
    print("Confusion matrix:\n", cm)

    # ---- plots for evaluation
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["healthy", "failure"], yticklabels=["healthy", "failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (test period)")   # lamba model naam title me nahi, warna plot me kat jaata hai
    save_fig("08_confusion_matrix.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, label=name)
        pr, rc, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rc, pr, label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=7)
    save_fig("09_roc_pr_curves.png")

    # ---- feature importance (permutation importance, kisi bhi model par chalta hai)
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(primary, X_test, y_test, scoring="average_precision",
                                  n_repeats=5, random_state=config.RANDOM_STATE)
    imp = pd.Series(perm.importances_mean, index=features).sort_values()
    plt.figure(figsize=(7, 6))
    imp.plot(kind="barh")
    plt.title("Feature importance (permutation, PR-AUC drop)")
    save_fig("10_feature_importance.png")
    top_features = imp.sort_values(ascending=False).head(5).index.tolist()
    print("Top features:", top_features)

    # ---- 5. anomaly detector (sirf healthy training data par seekhta hai)
    print("\n[ANOMALY] Isolation Forest")
    healthy_train = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=200, contamination=0.02,
                          random_state=config.RANDOM_STATE)
    iso.fit(healthy_train)
    anomaly_flag = (iso.predict(X_test) == -1).astype(int)
    an_prec = precision_score(y_test, anomaly_flag, zero_division=0)
    an_rec = recall_score(y_test, anomaly_flag, zero_division=0)
    print(f"  anomalies flagged: {anomaly_flag.sum()} / {len(anomaly_flag)} "
          f"| overlap with real failures -> precision {an_prec:.3f}, recall {an_rec:.3f}")

    # ---- 6. failure type classifier
    # Ye model sirf failure rows par train aur test hota hai. "No Failure" rows shamil
    # karte to majority class ki wajah se score jhootha achha dikhta. Prediction ke time
    # pehle failure model risk batata hai, phir ye batata hai ki failure aaye to kaun sa hoga.
    print("\n[FAILURE TYPE] Random Forest multi-class (failure rows only)")
    fail_train = train[train[config.COL_FAILURE] == 1]
    fail_test = test[test[config.COL_FAILURE] == 1]
    type_model = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                        n_jobs=-1, random_state=config.RANDOM_STATE)
    type_model.fit(fail_train[features], fail_train[config.COL_FAILURE_TYPE].values)
    type_pred = type_model.predict(fail_test[features])
    type_report = classification_report(fail_test[config.COL_FAILURE_TYPE].values, type_pred,
                                        zero_division=0, output_dict=True)
    failure_type_metrics = {
        "train_failures": int(len(fail_train)),
        "test_failures": int(len(fail_test)),
        "accuracy": float((type_pred == fail_test[config.COL_FAILURE_TYPE].values).mean()),
        "weighted_f1": float(type_report["weighted avg"]["f1-score"]),
        "macro_f1": float(type_report["macro avg"]["f1-score"]),
        # har failure type ka test me kitna support tha (kam support = score par kam bharosa)
        "support": fail_test[config.COL_FAILURE_TYPE].value_counts().to_dict(),
    }
    print(f"  train failures: {failure_type_metrics['train_failures']} | test failures: "
          f"{failure_type_metrics['test_failures']}")
    print(f"  accuracy {failure_type_metrics['accuracy']:.3f} | weighted F1 "
          f"{failure_type_metrics['weighted_f1']:.3f} | macro F1 {failure_type_metrics['macro_f1']:.3f}")
    print(classification_report(fail_test[config.COL_FAILURE_TYPE].values, type_pred, zero_division=0))

    # failure type confusion matrix (sirf test ke failure rows)
    type_labels = sorted(set(fail_test[config.COL_FAILURE_TYPE]) | set(type_pred))
    cm_type = confusion_matrix(fail_test[config.COL_FAILURE_TYPE].values, type_pred, labels=type_labels)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm_type, annot=True, fmt="d", cmap="Greens",
                xticklabels=[t.replace(" Failure", "") for t in type_labels],
                yticklabels=[t.replace(" Failure", "") for t in type_labels])
    plt.xlabel("Predicted type")
    plt.ylabel("Actual type")
    plt.title("Failure-type confusion matrix (test failures only)")
    save_fig("11_failure_type_confusion.png")

    # ---- 7. RUL regressor
    print("\n[RUL] Random Forest Regressor")
    rul_model = RandomForestRegressor(n_estimators=150, min_samples_leaf=5,
                                      n_jobs=-1, random_state=config.RANDOM_STATE)
    rul_model.fit(X_train, train["rul"].values)
    rul_pred = rul_model.predict(X_test)
    rul_metrics = {
        "mae": float(mean_absolute_error(test["rul"], rul_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test["rul"], rul_pred))),
        "r2": float(r2_score(test["rul"], rul_pred)),
        # seedha average predict karne wale "dumb" model ka MAE, comparison ke liye
        "baseline_mae": float(mean_absolute_error(
            test["rul"], np.full(len(test), train["rul"].mean()))),
    }
    print(f"  MAE={rul_metrics['mae']:.2f} (mean-baseline MAE={rul_metrics['baseline_mae']:.2f}) "
          f"RMSE={rul_metrics['rmse']:.2f} R2={rul_metrics['r2']:.3f}")
    print("  Note: RUL yahan experimental hai, accuracy limited hai.")

    # ---- 8. save artifact
    # Production ke liye primary + baki models ek hi file me. Pehle train data se
    # fit kiya tha (honest evaluation); ab primary ko poore data par dobara fit
    # karte hain taaki deployed model ke paas zyada data ho.
    X_all, y_all = df[features], df[config.COL_FAILURE].values
    final_model, _ = make_boosted_classifier(float((y_all == 0).sum() / max(1, (y_all == 1).sum())))
    final_model.fit(X_all, y_all)

    artifact = {
        "model": final_model,
        "features": features,
        "version": config.MODEL_VERSION,
        "algorithm": primary_name,
        "baseline": X_all.median().to_dict(),     # explanation ke liye "normal" values
        # drift monitoring ke liye training data ka mean/std (core sensors)
        "feature_stats": {
            c: {"mean": float(df[c].mean()), "std": float(df[c].std())}
            for c in [config.COL_TEMP, config.COL_SPEED, config.COL_TORQUE, config.COL_WEAR]
        },
        "anomaly_model": iso,
        "failure_type_model": type_model,
        "rul_model": rul_model,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # app load karte waqt check karti hai: alag scikit-learn version se bani file par
        # pickle chal to jaati hai par predictions bigad sakti hain, isliye tab dobara train hota hai
        "sklearn_version": sklearn.__version__,
    }
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(artifact, config.MODEL_PATH)
    print(f"\nModel saved: {config.MODEL_PATH}")

    # ---- 9. metrics json (dashboard par dikhane ke liye)
    metrics = {
        "algorithm": primary_name,
        "version": config.MODEL_VERSION,
        "trained_at": artifact["trained_at"],
        "split": {"cutoff": str(cutoff), "train_rows": int(len(train)), "test_rows": int(len(test))},
        "comparison": results,
        "primary": evaluate(primary_name, y_test, prob_primary),
        "confusion_matrix": cm.tolist(),
        "top_features": top_features,
        "anomaly": {"flagged": int(anomaly_flag.sum()), "precision": float(an_prec), "recall": float(an_rec)},
        "failure_type": failure_type_metrics,
        "failure_type_weighted_f1": failure_type_metrics["weighted_f1"],   # purana key, compatibility ke liye
        "rul": rul_metrics,
    }
    os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)
    with open(config.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)
    comparison.to_csv(os.path.join(os.path.dirname(config.METRICS_PATH), "model_comparison.csv"), index=False)

    print(f"Metrics saved: {config.METRICS_PATH}")
    print(f"Done in {time.time() - start:.1f} sec")


if __name__ == "__main__":
    main()
