import json
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

def preprocess_data(records):
    """Convert JSON-based medical appeal records into a structured DataFrame."""
    df = pd.DataFrame(records)
    
    # Extract top-level attributes
    df["gender"] = df["patient_info"].apply(lambda x: x["gender"])
    df["age_range"] = df["patient_info"].apply(lambda x: x["age_range"])
    df["disease"] = df["diagnosis"]
    df["treatment_category"] = df["treatment_category"]
    df["treatment_subcategory"] = df["treatment_subcategory"]
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
medical_appeals = []
cache_dir = "cache"

for json_file in glob.glob(os.path.join(cache_dir, "*.json")):
    with open(json_file, 'r') as f:
        medical_appeals.append(json.load(f))

print(f"Loaded {len(medical_appeals)} medical appeal records")

# Preprocess data
df = preprocess_data(medical_appeals)

# Prepare features and target
X = df.drop(columns=["is_denial_upheld"])
y = df["is_denial_upheld"]

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Get columns that have non-null values after imputation
non_null_cols = [col for col, is_null in zip(X.columns, np.any(np.isnan(X_imputed), axis=0)) if not is_null]
X = pd.DataFrame(X_imputed, columns=non_null_cols)

# Dropped columns
dropped_cols = set(X.columns) - set(non_null_cols)
if dropped_cols:
    print(f"Columns dropped due to all null values: {dropped_cols}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TPOT with Linear Regression only
tpot = TPOTRegressor(
    generations=5,
    population_size=20,
    verbosity=2,
    template='Regressor',  # Use the Regressor template
    regressor=['LinearRegression'],  # Restrict to only LinearRegression
    random_state=42
)
tpot.fit(X_train, y_train)

# Get the best model
best_model = tpot.fitted_pipeline_

# Evaluate model performance
r2_score = tpot.score(X_test, y_test)
print(f"Best Model R^2 Score: {r2_score:.4f}")

# Extract linear regression coefficients
lr_model = best_model.steps[-1][1]  # Get final step of pipeline
coefficients = lr_model.coef_
feature_names = X.columns

# Create a DataFrame for feature importance
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

# Show most positive and most negative predictors
print("Most Positive Predictors:")
print(coef_df.head(10))
print("\nMost Negative Predictors:")
print(coef_df.tail(10))

# Plot coefficients
plt.figure(figsize=(10, 6))
plt.barh(coef_df["Feature"], coef_df["Coefficient"])
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title("Linear Regression Feature Importance in Medical Appeals Prediction")
plt.show()

# Export best pipeline
tpot.export('best_model_pipeline.py')