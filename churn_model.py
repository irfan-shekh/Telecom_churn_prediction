"""
Production-Grade Machine Learning Pipeline for Telecom Customer Churn Prediction.

Features:
- Robust data ingestion and schema validation
- Reproducible, zero-leakage feature engineering
- Scikit-Learn Pipeline encapsulation (StandardScaler + OneHotEncoder + Classifier)
- Model comparison & selection (XGBoost, Random Forest, Logistic Regression)
- Production artifact serialization (Pipeline + Metadata JSON)
- CLI interface with structured logging
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ChurnModelPipeline")

# Feature definition contracts
NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlyCharges",
    "Charge_Drift",
    "TotalServicesCount",
    "Protection_Security_Score",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "AutoPayment",
    "Is_Long_Term_Contract",
    "Streaming_Only_Risk",
    "Household_Type",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load and validate dataset from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    logger.info(f"Loading raw dataset from {filepath}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data and derives domain-specific features without data leakage.
    Can be applied on both training batches and single prediction payloads.
    """
    data = df.copy()

    # 1. TotalCharges cleaning & rectification
    if "TotalCharges" in data.columns:
        if data["TotalCharges"].dtype == object:
            data["TotalCharges"] = (
                pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
                .fillna(0.0)
            )
        else:
            data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    # 2. Total active services count (Product Stickiness)
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    
    total_services = pd.Series(0, index=data.index)
    for col in service_cols:
        if col in data.columns:
            total_services += (data[col] == "Yes").astype(int)
    
    if "InternetService" in data.columns:
        total_services += data["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)

    data["TotalServicesCount"] = total_services

    # 3. Security & tech support moat score (0-4)
    moat_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    moat_score = pd.Series(0, index=data.index)
    for col in moat_cols:
        if col in data.columns:
            moat_score += (data[col] == "Yes").astype(int)
    data["Protection_Security_Score"] = moat_score

    # 4. Financial trajectory & bill shock indicators
    tenure_safe = data["tenure"].astype(float) + 1.0
    data["AvgMonthlyCharges"] = data["TotalCharges"] / tenure_safe
    data["Charge_Drift"] = data["MonthlyCharges"] - data["AvgMonthlyCharges"]

    # 5. Frictionless billing flag
    if "PaymentMethod" in data.columns:
        data["AutoPayment"] = data["PaymentMethod"].apply(
            lambda x: 1 if "automatic" in str(x).lower() else 0
        )
    else:
        data["AutoPayment"] = 0

    # 6. Contract commitment flag
    if "Contract" in data.columns:
        data["Is_Long_Term_Contract"] = data["Contract"].apply(
            lambda x: 1 if x in ["One year", "Two year"] else 0
        )
    else:
        data["Is_Long_Term_Contract"] = 0

    # 7. Entertainment without support risk flag
    has_streaming = (data.get("StreamingTV") == "Yes") | (data.get("StreamingMovies") == "Yes")
    no_security = (data.get("OnlineSecurity") != "Yes") & (data.get("TechSupport") != "Yes")
    data["Streaming_Only_Risk"] = (has_streaming & no_security).astype(int)

    # 8. Household stability segment
    def compute_household(row: pd.Series) -> str:
        partner = row.get("Partner") == "Yes"
        dependents = row.get("Dependents") == "Yes"
        if partner and dependents:
            return "Family"
        elif partner:
            return "Couple"
        elif dependents:
            return "Single_Parent"
        return "Solo"

    data["Household_Type"] = data.apply(compute_household, axis=1)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles numeric normalization
    and categorical one-hot encoding with out-of-vocabulary protection.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore",
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_model(model_name: str, random_state: int = 42) -> Any:
    """Factory function for instantiating model classifiers."""
    model_name_lower = model_name.lower().replace(" ", "").replace("_", "")

    if "xgb" in model_name_lower:
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        )
    elif "randomforest" in model_name_lower or "rf" in model_name_lower:
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        )
    elif "logistic" in model_name_lower or "lr" in model_name_lower:
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: 'xgboost', 'random_forest', 'logistic_regression'."
        )


def build_pipeline(model_name: str = "xgboost", random_state: int = 42) -> Pipeline:
    """Builds an end-to-end production pipeline."""
    preprocessor = build_preprocessor()
    classifier = get_model(model_name, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def evaluate_model(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Generates comprehensive validation metrics for production tracking."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = float(roc_auc_score(y_test, y_proba))

    metrics = {
        "accuracy": float(cr["accuracy"]),
        "precision_churn": float(cr["1"]["precision"]),
        "recall_churn": float(cr["1"]["recall"]),
        "f1_churn": float(cr["1"]["f1-score"]),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "test_samples": int(len(y_test)),
    }
    return metrics


def train_production_model(
    data_path: str = "Telco_Customer_Churn.csv",
    model_name: str = "xgboost",
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: str = ".",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Main training execution function.
    Trains, evaluates, and serializes the complete production pipeline artifact.
    """
    # 1. Load data
    df_raw = load_dataset(data_path)

    # 2. Feature engineering
    logger.info("Executing zero-leakage feature engineering...")
    df_engineered = engineer_features(df_raw)

    # 3. Prepare target and features
    if "customerID" in df_engineered.columns:
        df_engineered = df_engineered.drop(columns=["customerID"])

    if "Churn" not in df_engineered.columns:
        raise KeyError("Target column 'Churn' not found in dataset.")

    y = df_engineered["Churn"].map({"Yes": 1, "No": 0})
    X = df_engineered.drop(columns=["Churn"])

    # 4. Stratified Train-Test split
    logger.info(f"Splitting data with test_size={test_size}, stratify=y...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Build and train pipeline
    logger.info(f"Building pipeline with classifier: {model_name}...")
    pipeline = build_pipeline(model_name=model_name, random_state=random_state)

    logger.info("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    logger.info("Evaluating model performance on unseen test split...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    cm = metrics["confusion_matrix"]
    logger.info(
        f"Evaluation Results for [{model_name}]:\n"
        f"  - Accuracy:         {metrics['accuracy']:.4f}\n"
        f"  - Precision (Churn):{metrics['precision_churn']:.4f}\n"
        f"  - Recall (Churn):   {metrics['recall_churn']:.4f}\n"
        f"  - F1-Score (Churn): {metrics['f1_churn']:.4f}\n"
        f"  - ROC-AUC:          {metrics['roc_auc']:.4f}\n"
        f"  - Confusion Matrix: TN={cm['true_negatives']}, FP={cm['false_positives']}, "
        f"FN={cm['false_negatives']}, TP={cm['true_positives']}"
    )

    # 7. Serialize production artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_artifact_path = os.path.join(output_dir, "churn_pipeline.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")

    logger.info(f"Serializing pipeline artifact to {model_artifact_path}...")
    joblib.dump(pipeline, model_artifact_path)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": metrics,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "artifact_file": os.path.basename(model_artifact_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata to {metadata_path}.")

    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Production Machine Learning Training Pipeline for Telecom Customer Churn."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Telco_Customer_Churn.csv",
        help="Path to Telco Customer Churn CSV file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic_regression"],
        help="Algorithm to train (default: xgboost).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset reserved for testing (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save serialized model artifact and metadata.",
    )

    args = parser.parse_args()
    train_production_model(
        data_path=args.data_path,
        model_name=args.model_type,
        test_size=args.test_size,
        random_state=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
