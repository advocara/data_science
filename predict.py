import json
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def preprocess_data(records):
    """Convert JSON-based medical appeal records into a structured DataFrame."""
    df = pd.DataFrame(records)
    
    # Extract top-level attributes
    df["gender"] = df["patient_info"].apply(lambda x: x["gender"])
    df["age_range"] = df["patient_info"].apply(lambda x: x["age_range"])
    df["disease"] = df["diagnosis"]
    df["treatment_category"] = df["treatment_category"] # NOOP
    df["treatment_subcategory"] = df["treatment_subcategory"] # NOOP
    df["is_denial_upheld"] = df["is_denial_upheld"].astype(int)  # Target variable
    
    # Encode support flags (binary)
    df["guidelines_support"] = df["guidelines_support"].fillna(False).astype(int)
    df["guidelines_not_support"] = df["guidelines_not_support"].fillna(False).astype(int)
    df["soc_support"] = df["soc_support"].fillna(False).astype(int)
    df["soc_not_support"] = df["soc_not_support"].fillna(False).astype(int)

    # Encode treatments (one-hot encoding for dynamic values)
    def encode_treatments(col_name):
        all_treatments = set()
        for record in records:
            all_treatments.update([t["name"] for t in record.get(col_name, [])])
        for treatment in all_treatments:
            df[f"{col_name}_{treatment}"] = df[col_name].apply(lambda x: int(any(t["name"] == treatment for t in x)))
    
    encode_treatments("treatments_requested")
    encode_treatments("treatments_tried_but_failed")
    
    # Encode categorical variables
    categorical_cols = ["gender", "age_range", "disease", "treatment_category", "treatment_subcategory"]
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    
    # Drop original JSON columns
    df = df.drop(columns=["patient_info", "diagnosis", "secondary_conditions", "complications", "symptoms", 
                          "treatments_requested", "treatments_tried_but_failed", "treatments_tried_and_worked", 
                          "treatments_not_tried", "issues_considered", "guidelines_details", "soc_details", "study_details", 
                          "key_questions", "rationale", "reviewer_credentials", "case_id"], errors="ignore")
    
    return df

# Load data
# Load JSON files from cache directory
medical_appeals = []
cache_dir = "cache"

# Read all JSON files in cache directory
for json_file in glob.glob(os.path.join(cache_dir, "*.json")):
    with open(json_file, 'r') as f:
        medical_appeals.append(json.load(f))

print(f"Loaded {len(medical_appeals)} medical appeal records")


# Preprocess data
df = preprocess_data(medical_appeals)

# Split into train-test sets
X = df.drop(columns=["is_denial_upheld"])
y = df["is_denial_upheld"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Feature Importance
feature_importances = clf.feature_importances_
plt.barh(X.columns, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance in Medical Appeals Prediction")
plt.show()