import json
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from random_forest_paths import extract_multi_feature_paths
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
    
    # Encode support flags (binary) - be explicit about handling None values
    boolean_columns = [
        "guidelines_support",
        "guidelines_not_support", 
        "soc_support",
        "soc_not_support"
    ]
    
    for col in boolean_columns:
        # Convert None to False and ensure boolean type
        df[col] = df[col].map({True: 1, False: 0, None: 0}).astype(int)

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


def train_random_forest(X, y):
    """
    Train a Random Forest model on the given data.
    
    Args:
        X: Feature DataFrame
        y: Target variable Series
    
    Returns:
        tuple: Trained model, X_train, X_test, y_train, y_test
    """
    # Split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train RandomForest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, X_train, X_test, y_train, y_test

# Preprocess data
df = preprocess_data(medical_appeals)

# Prepare features and target
X = df.drop(columns=["is_denial_upheld"])
y = df["is_denial_upheld"]

# Train model
model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

# Feature Importance
feature_importances = model.feature_importances_
# plt.barh(X.columns, feature_importances)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.title("Random Forest Feature Importance in Medical Appeals Prediction")
# plt.show()


#### ======

# **Use our path counting module for Feature Path Analysis**
print("\n--- Most Common Multi-Feature Decision Paths ---")
top_paths = extract_multi_feature_paths(model, X, min_occurrences=2, max_path_length=4)
for path, count in top_paths.items():
    path_str = " + ".join(path)  # Convert tuple to readable format
    print(f"{path_str} (Appeared {count} times)")


# print("\n--- Most Common Multi-Feature Pathways ---")
# common_combinations = random_forests_paths.extract_feature_combinations(model, X.columns)
# for features, count in common_combinations.items():
#     print(f"{features} (Appeared {count} times)")

# # **Visualize a Sample Decision Tree**
# random_forests_paths.visualize_decision_tree(model, X.columns, max_depth=3)
