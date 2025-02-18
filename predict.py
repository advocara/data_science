import json
import glob
import os
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


from random_forest_paths import extract_multi_feature_paths, analyze_feature_impact, format_impact_analysis, FeatureImpact
from model.appeal import MedicalInsuranceAppeal
from term_normalizer import Normalizer

from run_FPGrowth import run_FP
from run_deap_genetic import run_deap


def appeals_as_dataframe_onehot(records):
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
    
    # Drop original JSON columns now that we have extracted the data to one-hot encoded columns
    df = df.drop(columns=["patient_info", "diagnosis", "secondary_conditions", "complications", "symptoms", 
                          "treatments_requested", "treatments_tried_but_failed", "treatments_tried_and_worked", 
                          "treatments_not_tried", "issues_considered", "guidelines_details", "soc_details", "study_details", 
                          "key_questions", "rationale", "reviewer_credentials", "case_id"], errors="ignore")
    
    return df


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
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
    clf.fit(X_train, y_train)
    
    return clf, X_train, X_test, y_train, y_test


# Load JSON files from cache directory
medical_appeals = []
normalized_appeal_dir = "D:\\Advocara\\data_science\\appeals-results"
category_substring = "Immuno Disorders-Lupus-norm"

# Read all JSON files in cache directory
for json_file in glob.glob(os.path.join(normalized_appeal_dir, f"*{category_substring}*.json")):
    with open(json_file, 'r') as f:
        medical_appeals.append(MedicalInsuranceAppeal.model_validate(json.load(f)))

print(f"Loaded {len(medical_appeals)} medical appeal records for: {category_substring}")

#### Convert to a dataframe with categories flattened as one-hot encoded columns ####
medical_appeal_dicts = [appeal.model_dump() for appeal in medical_appeals]
df = appeals_as_dataframe_onehot(medical_appeal_dicts) # convert back to dict now that name normalization is done

method = "deap"

if method == "random_forest":
    #### Prepare features and target ####
    X = df.drop(columns=["is_denial_upheld"])
    y = df["is_denial_upheld"]
    # Train model
    model, X_train, X_test, y_train, y_test = train_random_forest(X, y)

    # Feature Importance
    feature_importances = model.feature_importances_

    ### Plot single-feature importance and display bar chart ###
    # plt.barh(X.columns, feature_importances)
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Feature")
    # plt.title("Random Forest Feature Importance in Medical Appeals Prediction")
    # plt.show()
    #### ======

    # **Use our path counting module for Feature Path Analysis**
    print("\n\n\n\n\n===\n===\n=== Feature Impact Analysis ===")
    impact_metrics: Dict[Tuple[str, ...], FeatureImpact] = analyze_feature_impact(model, X, min_occurrences=5, max_path_length=3)
    results = format_impact_analysis(impact_metrics, top_n=50)

    print("\nMost Impactful Feature Combinations:")
    print("------------------------------------")
    for result in results:
        print(result)
        print("------------------------------------")

    # print("\n--- Most Common Multi-Feature Pathways ---")
    # common_combinations = random_forests_paths.extract_feature_combinations(model, X.columns)
    # for features, count in common_combinations.items():
    #     print(f"{features} (Appeared {count} times)")

    # # **Visualize a Sample Decision Tree**
    # random_forests_paths.visualize_decision_tree(model, X.columns, max_depth=3)



elif method == 'FPgrowth':
    run_FP(df)

elif method == "deap":
    # for col in df.columns:
    #     print(col)

    # print(df[(df['guidelines_support'] == 0) & (df['guidelines_not_support'] == 0)])
    df['soc_absent'] = ((df['soc_support'] == 0) & (df['soc_not_support'] == 0)).astype(int)
    df['guidelines_absent'] = ((df['guidelines_support'] == 0) & (df['guidelines_not_support'] == 0)).astype(int)
    run_deap(df, 'deap_results_support_absent_elitism_new.csv')