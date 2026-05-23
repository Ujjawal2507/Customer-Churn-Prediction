"""
Customer Churn Prediction Pipeline v2
=======================================
Improvements over v1:
- Class imbalance handling via SMOTE + class_weight
- XGBoost added as 4th model
- Threshold optimization for recall
- GridSearchCV scoring on recall
- Precision-Recall curve saved
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, recall_score
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[!] imbalanced-learn not installed. Skipping SMOTE. Run: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[!] XGBoost not installed. Skipping XGB. Run: pip install xgboost")


# ─── 1. DATA LOADING ────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[✓] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def quick_summary(df: pd.DataFrame):
    print("\n── Dataset Summary ──────────────────────────────────")
    missing = df.isnull().sum()
    print("Missing values:", missing[missing > 0].to_dict() if missing.any() else "None")
    print("\n── Churn Distribution ───────────────────────────────")
    print(df["Churn"].value_counts())
    print(df["Churn"].value_counts(normalize=True).map("{:.1%}".format))


# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in ["customerID", "Churn"]:
        if col in cat_cols:
            cat_cols.remove(col)

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Encode target
    if df["Churn"].dtype == object:
        df["Churn"] = le.fit_transform(df["Churn"].astype(str))

    print(f"[✓] Preprocessed: {len(cat_cols)} categorical columns encoded")
    return df


def split_scale(df: pd.DataFrame, target: str = "Churn", test_size: float = 0.2):
    X = df.drop(columns=["customerID", target], errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"[✓] Split → Train: {X_train_s.shape[0]:,}  Test: {X_test_s.shape[0]:,}")
    return X_train_s, X_test_s, y_train, y_test, scaler, X.columns.tolist()


# ─── 3. CLASS IMBALANCE — SMOTE ──────────────────────────────────────────────

def apply_smote(X_train, y_train):
    if not SMOTE_AVAILABLE:
        print("[!] SMOTE skipped — imbalanced-learn not installed")
        return X_train, y_train

    before = dict(pd.Series(y_train).value_counts())
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    after = dict(pd.Series(y_res).value_counts())
    print(f"[✓] SMOTE: {before} → {after}")
    return X_res, y_res


# ─── 4. MODEL DEFINITIONS ────────────────────────────────────────────────────

def get_models(pos_weight: float = 1.0) -> dict:
    models = {
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
    return models


# ─── 5. THRESHOLD OPTIMIZATION ───────────────────────────────────────────────

def find_best_threshold(y_test, y_proba, min_recall: float = 0.80) -> float:
    """Find the lowest threshold that achieves at least min_recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    viable = thresholds[recalls[:-1] >= min_recall]
    if len(viable) == 0:
        print(f"[!] Could not achieve {min_recall:.0%} recall. Using 0.35.")
        return 0.35
    best = viable[0]
    print(f"[✓] Best threshold for ≥{min_recall:.0%} recall: {best:.3f}")
    return float(best)


# ─── 6. TRAINING ─────────────────────────────────────────────────────────────

def train_all(X_train, y_train) -> dict:
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(pos_weight)
    fitted = {}

    print("\n── Training models ──────────────────────────────────")
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring="recall").mean()
        cv_auc    = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
        print(f"  {name:<25} CV Recall: {cv_recall:.4f}   CV AUC: {cv_auc:.4f}")
        fitted[name] = model

    return fitted


# ─── 7. EVALUATION ───────────────────────────────────────────────────────────

def evaluate(models: dict, X_test, y_test, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n── Evaluation (default threshold = 0.5) ─────────────")
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_proba)
        rec     = recall_score(y_test, y_pred)
        results[name] = {"accuracy": acc, "auc": auc, "recall": rec,
                         "y_pred": y_pred, "y_proba": y_proba}
        print(f"\n  {name}")
        print(f"  Accuracy={acc:.4f}  AUC={auc:.4f}  Recall={rec:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_proba = results[best_name]["y_proba"]

    # ── Optimized threshold ──
    print(f"\n── Threshold optimization for {best_name} ───────────")
    best_thresh = find_best_threshold(y_test, best_proba, min_recall=0.80)
    y_pred_opt  = (best_proba >= best_thresh).astype(int)
    rec_opt     = recall_score(y_test, y_pred_opt)
    acc_opt     = accuracy_score(y_test, y_pred_opt)
    print(f"  Optimized → Accuracy: {acc_opt:.4f}  Recall: {rec_opt:.4f}")
    print(classification_report(y_test, y_pred_opt, target_names=["No Churn", "Churn"]))

    results[best_name]["y_pred_opt"]  = y_pred_opt
    results[best_name]["best_thresh"] = best_thresh

    _plot_confusion_matrix(y_test, y_pred_opt, best_name, output_dir)
    _plot_roc_curves(results, y_test, output_dir)
    _plot_precision_recall(y_test, best_proba, best_name, best_thresh, output_dir)

    return results, best_name, best_thresh


def _plot_confusion_matrix(y_test, y_pred, name, output_dir):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name} (optimized threshold)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[✓] Saved → {output_dir}/confusion_matrix.png")


def _plot_roc_curves(results, y_test, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/roc_curves.png", dpi=150)
    plt.close()
    print(f"[✓] Saved → {output_dir}/roc_curves.png")


def _plot_precision_recall(y_test, y_proba, name, threshold, output_dir):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions[:-1], label="Precision", color="#378ADD")
    ax.plot(thresholds, recalls[:-1],    label="Recall",    color="#D85A30")
    ax.axvline(threshold, color="gray", linestyle="--", lw=1.2,
               label=f"Chosen threshold ({threshold:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision–Recall Tradeoff — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"[✓] Saved → {output_dir}/precision_recall_curve.png")


# ─── 8. FEATURE IMPORTANCE ───────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, output_dir: str = "outputs", top_n: int = 15):
    if not hasattr(model, "feature_importances_"):
        print("[!] Model has no feature_importances_. Skipping.")
        return

    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    plt.close()
    print(f"[✓] Saved → {output_dir}/feature_importance.png")


# ─── 9. EXPORT ───────────────────────────────────────────────────────────────

def export_artifacts(model, scaler, feature_names: list, threshold: float, output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model,   f"{output_dir}/best_model.pkl")
    joblib.dump(scaler,  f"{output_dir}/scaler.pkl")
    pd.Series(feature_names).to_csv(f"{output_dir}/features.csv", index=False, header=["feature"])
    with open(f"{output_dir}/threshold.txt", "w") as f:
        f.write(str(threshold))
    print(f"[✓] Exported model, scaler, features, threshold → {output_dir}/")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH  = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OUTPUT_DIR = "outputs"

    print("=" * 55)
    print("  Customer Churn Prediction Pipeline v2")
    print("=" * 55)

    df       = load_data(DATA_PATH)
    quick_summary(df)

    df_clean = preprocess(df)
    X_train, X_test, y_train, y_test, scaler, features = split_scale(df_clean)

    # Apply SMOTE to training set
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Train
    models = train_all(X_train_bal, y_train_bal)

    # Evaluate
    results, best_name, best_thresh = evaluate(models, X_test, y_test, OUTPUT_DIR)

    # Feature importance
    plot_feature_importance(models[best_name], features, OUTPUT_DIR)

    # Export
    export_artifacts(models[best_name], scaler, features, best_thresh, OUTPUT_DIR)

    print(f"\n{'='*55}")
    print(f"  Best model : {best_name}")
    print(f"  AUC        : {results[best_name]['auc']:.4f}")
    print(f"  Threshold  : {best_thresh:.3f}")
    print(f"{'='*55}")
