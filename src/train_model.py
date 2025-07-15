import pandas as pd
import numpy as np
import yaml
import mlflow
import joblib
import os

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

from sklearn.model_selection import train_test_split

print("Loading params from params.yaml...")
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Paths
processed_path = params["data"]["processed_data_path"]
target_col = params["data"]["target_column"]

print("Reading processed train/test CSV files...")
X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).values.ravel()

print(f"Data Loaded: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Categorical and numerical columns
categorical_cols = ['category', 'gender', 'state', 'job', 'merchant']
numerical_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time',
                  'time_since_last_transaction', 'merch_lat', 'merch_long',
                  'hour_of_day', 'day_of_week', 'month']

print("Setting up preprocessing pipeline...")
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

print("Configuring Random Forest Classifier pipeline...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=params['model']['random_forest']['n_estimators'],
        max_depth=params['model']['random_forest']['max_depth'],
        random_state=params['model']['random_forest']['random_state'],
        class_weight=params['model']['random_forest']['class_weight']
    ))
])

print("Applying Isolation Forest on test set to identify anomalies...")
iso_forest = IsolationForest(
    n_estimators=params['model']['isolation_forest']['n_estimators'],
    contamination=params['model']['isolation_forest']['contamination'],
    random_state=params['model']['isolation_forest']['random_state'],
    n_jobs=-1
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

iso_forest.fit(X_train_proc)
anomaly_scores = -iso_forest.decision_function(X_test_proc)
threshold = np.percentile(anomaly_scores, 95)
mask_suspects = anomaly_scores >= threshold

X_test_suspects = X_test.iloc[mask_suspects]
y_test_suspects = y_test[mask_suspects]
print(f"Selected {sum(mask_suspects)} suspected fraudulent samples from test set.")

print("Training Random Forest classifier on full training data...")
rf_pipeline.fit(X_train, y_train)

print("Predicting probabilities on suspected fraud samples...")
y_prob = rf_pipeline.predict_proba(X_test_suspects)[:, 1]
cutoff = params["threshold"]["probability_cutoff"]
y_pred = (y_prob >= cutoff).astype(int)

print("Calculating performance metrics...")
acc = accuracy_score(y_test_suspects, y_pred)
prec = precision_score(y_test_suspects, y_pred, pos_label=1)
rec = recall_score(y_test_suspects, y_pred, pos_label=1)
f1 = f1_score(y_test_suspects, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test_suspects, y_prob)
pr_auc = average_precision_score(y_test_suspects, y_prob)

print("\nClassification Report:")
print(classification_report(y_test_suspects, y_pred))

print("Logging experiment to MLflow...")
mlflow.set_experiment("fraud_detection_hybrid")

with mlflow.start_run():
    mlflow.log_params(params["model"]["random_forest"])
    mlflow.log_param("probability_cutoff", cutoff)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    })

    model_path = "models/best_model.pkl"
    joblib.dump(rf_pipeline, model_path)
    mlflow.log_artifact(model_path)

    print("Model training completed and logged to MLflow.")
    print(f"Best model saved at: {model_path}")
