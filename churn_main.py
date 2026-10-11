"""
Main training execution script for Telecom Customer Churn Prediction.
Delegates to the production-grade pipeline in churn_model.py.
"""

from churn_model import train_production_model

if __name__ == "__main__":
    print("--- Starting Production Churn Model Training ---")
    pipeline, metrics = train_production_model(
        data_path="Telco_Customer_Churn.csv",
        model_name="xgboost",
        test_size=0.20,
        random_state=42,
        output_dir=".",
    )
    print("\n--- Training Completed Successfully ---")
    print(f"ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"Accuracy:         {metrics['accuracy']:.4f}")
    print(f"Precision (Churn):{metrics['precision_churn']:.4f}")
    print(f"Recall (Churn):   {metrics['recall_churn']:.4f}")
    print(f"F1-Score (Churn): {metrics['f1_churn']:.4f}")
    print("Serialized Pipeline: churn_pipeline.joblib")
    print("Serialized Metadata: model_metadata.json")
