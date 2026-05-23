"
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


def load_artifacts(model_path: str, scaler_path: str, features_path: str):
    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    features = pd.read_csv(features_path)["feature"].tolist()
    return model, scaler, features


def preprocess_input(df: pd.DataFrame, feature_names: list) -> np.ndarray:
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

    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]


def predict(input_path: str, output_path: str,
            model_path="outputs/best_model.pkl",
            scaler_path="outputs/scaler.pkl",
            features_path="outputs/features.csv"):

    model, scaler, features = load_artifacts(model_path, scaler_path, features_path)

    df_raw  = pd.read_csv(input_path)
    ids     = df_raw["customerID"] if "customerID" in df_raw.columns else pd.RangeIndex(len(df_raw))

    X       = preprocess_input(df_raw, features)
    X_s     = scaler.transform(X)

    probs   = model.predict_proba(X_s)[:, 1]
    labels  = model.predict(X_s)

    result = pd.DataFrame({
        "customerID":     ids,
        "churn_prob":     probs.round(4),
        "churn_pred":     labels,
        "risk_segment":   pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                 labels=["Low", "Medium", "High"]),
    })

    result.to_csv(output_path, index=False)
    print(f"[✓] Predictions written to {output_path}")
    print(result["risk_segment"].value_counts())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Inference")
    parser.add_argument("--input",   required=True,  help="Path to new customers CSV")
    parser.add_argument("--output",  default="outputs/predictions.csv")
    parser.add_argument("--model",   default="outputs/best_model.pkl")
    parser.add_argument("--scaler",  default="outputs/scaler.pkl")
    parser.add_argument("--features",default="outputs/features.csv")
    args = parser.parse_args()

    predict(args.input, args.output, args.model, args.scaler, args.features)
