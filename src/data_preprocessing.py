import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

# Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

RAW_PATH = params["data"]["raw_data_path"]
PROCESSED_PATH = params["data"]["processed_data_path"]
TARGET = params["data"]["target_column"]
TEST_SIZE = params["split"]["test_size"]
RANDOM_STATE = params["split"]["random_state"]

# Ensure output folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load data
df = pd.read_csv(RAW_PATH)

# --- FEATURE ENGINEERING BASED ON YOUR NOTEBOOK ---

# Convert transaction time to datetime
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour_of_day"] = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["month"] = df["trans_date_trans_time"].dt.month

# Sort transactions and compute time since last transaction per card
df = df.sort_values(by=["cc_num", "trans_date_trans_time"])
df["time_since_last_transaction"] = df.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds().fillna(0)

# Drop irrelevant columns
columns_to_drop = ['Unnamed: 0', 'trans_num', 'first', 'last', 'street', 'city', 'zip', 'dob', 'ssn']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Drop original datetime field
df = df.drop(columns=["trans_date_trans_time"], errors='ignore')

# Split features and target
X = df.drop(columns=[TARGET], errors='ignore')
y = df[TARGET]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Save splits
X_train.to_csv(f"{PROCESSED_PATH}/X_train.csv", index=False)
X_test.to_csv(f"{PROCESSED_PATH}/X_test.csv", index=False)
y_train.to_csv(f"{PROCESSED_PATH}/y_train.csv", index=False)
y_test.to_csv(f"{PROCESSED_PATH}/y_test.csv", index=False)

print(" Data preprocessing completed and saved in data/processed/")
