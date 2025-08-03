# preprocessor.py
import pandas as pd
import joblib
import numpy as np
import os

ENCODERS_DIR = "encoders"
FEATURE_NAMES_PATH = os.path.join(ENCODERS_DIR, "feature_names.pkl")

# Columns to drop before prediction
COLUMNS_TO_DROP = [
    'id', 'manager.name', 'input.type', 'rule.groups', 'rule.pci_dss', 'rule.nist_800_53',
    'rule.tsc', 'rule.gdpr', 'rule.mitre.technique', 'rule.mitre.id', 'rule.mitre.tactic',
    'rule.level', 'rule.firedtimes', 'rule.info', 'data.url'
]


def parse_timestamp_features(df, col='@timestamp'):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df['hour'] = df[col].dt.hour
    df['weekday'] = df[col].dt.weekday
    df['month'] = df[col].dt.month
    df = df.drop(columns=[col])
    return df


def encode_categoricals(df):
    for col in df.select_dtypes(include=['object']).columns:
        encoder_path = os.path.join(ENCODERS_DIR, f"{col}_encoder.pkl")
        if os.path.exists(encoder_path):
            le = joblib.load(encoder_path)
            df[col] = df[col].astype(str).fillna("Missing")
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Missing")
            if "Missing" not in le.classes_:
                le.classes_ = np.append(le.classes_, "Missing")
            df[col] = le.transform(df[col])
        else:
            # Drop unknown object columns
            df = df.drop(columns=[col])
    return df


def preprocess(alert: dict) -> list:
    df = pd.DataFrame([alert])

    # Drop unused fields
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

    # Timestamp engineering
    if '@timestamp' in df.columns:
        df = parse_timestamp_features(df)

    # Encode object/categorical
    df = encode_categoricals(df)

    # Load feature names
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Align column order
    df = df[feature_names]

    # Fill remaining NaNs
    df = df.fillna(0)

    return df.iloc[0].tolist()
