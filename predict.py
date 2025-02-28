import json
import glob
import os
from typing import Dict, Tuple
import hashlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


from imr_analyzer import IMRAnalyzer
from model.appeal import MedicalInsuranceAppeal
from model.config import IMRConfig
from random_forest import run_random_forest
from term_normalizer import Normalizer

from run_FPGrowth import run_FP
from deap_genetic import run_deap


def appeals_as_dataframe_onehot(records):
    """Convert JSON-based medical appeal records into a structured DataFrame."""
    df = pd.DataFrame(records)
    
    # Extract top-level attributes
    df["gender"] = df["patient_info"].apply(lambda x: x["gender"])
    df["age_range"] = df["patient_info"].apply(lambda x: x["age_range"])
    df["disease"] = df["diagnosis"]
    
    # Combine treatment int one column
    df["treatment"] = df.apply(lambda x: f"{x['treatment_category']}_{x['treatment_subcategory']}", axis=1)
    df.drop(columns=["treatment_category", "treatment_subcategory"], inplace=True)

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
    encode_treatments("treatments_tried_and_worked")
    encode_treatments("treatments_not_tried")
    
    # Encode categorical variables
    categorical_cols = ["gender", "age_range", "disease", "treatment"]
    df = pd.get_dummies(df, columns=categorical_cols, dtype=int)
    
    # Drop original JSON columns now that we have extracted the data to one-hot encoded columns
    columns_to_drop = [
        "patient_info", "diagnosis", "secondary_conditions", "complications", 
        "symptoms", "treatments_requested", "treatments_tried_but_failed", 
        "treatments_tried_and_worked", "treatments_not_tried", "issues_considered", 
        "guidelines_details", "soc_details", "study_details", "key_questions", 
        "rationale", "reviewer_credentials", "case_id", "treatment"
    ]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    
    # soc or guidelines present are not that useful, so remove. Those are mentioned in text when they justify the overturn decision, but left out otherwise.
    # meaning the flag is a proxy for uphold/overturn and not an indicator of whether or not soc/guidelines were in the original appeal.
    df['soc_absent'] = ((df['soc_support'] == 0) & (df['soc_not_support'] == 0)).astype(int) # no mention of soc
    df['guidelines_absent'] = ((df['guidelines_support'] == 0) & (df['guidelines_not_support'] == 0)).astype(int) # no mention of guidelines
    
    return df


def predict(config: IMRConfig):
    # Load JSON files from cache directory
    medical_appeals = []
    normalized_appeal_dir = os.getcwd()+"/appeals-results"

    # Use IMRConfig method to generate the matching substring
    category_substring = config.generate_match_substring()

    # Read all JSON files in cache directory
    for json_file in glob.glob(os.path.join(normalized_appeal_dir, f"*{category_substring}*.json")):
        with open(json_file, 'r') as f:
            medical_appeals.append(MedicalInsuranceAppeal.model_validate(json.load(f)))

    print(f"Loaded {len(medical_appeals)} medical appeal records for: {category_substring}")

    #### Convert to a dataframe with categories flattened as one-hot encoded columns ####
    medical_appeal_dicts = [appeal.model_dump() for appeal in medical_appeals]
    df = appeals_as_dataframe_onehot(medical_appeal_dicts) # convert back to dict now that name normalization is done

    # Consolidate SOC/Guidelines flags 
    df['standards_followed'] = ((df['soc_support'] == 1) | (df['guidelines_support'] == 1)).astype(int)
    df['standards_not_mentioned'] = ((df['soc_absent'] == 1) & (df['guidelines_absent'] == 1)).astype(int)
    df['standards_not_followed'] = ((df['soc_not_support'] == 1) | (df['guidelines_not_support'] == 1)).astype(int)
    df.drop(columns=['soc_support', 'soc_not_support', 'guidelines_support', 'guidelines_not_support', 'soc_absent', 'guidelines_absent'], inplace=True)
        
    # Split data into three subsets. Each is an interesting subset of the data.
    # df_not_mentioned: standards not mentioned in the appeal - most important since appeals are tricky without standards.
    # df_present/absent: extra checks to confirm anomolies where standards do not dictate the outcome.
    df_standards_not_mentioned = df[(df['standards_not_mentioned'] == 1)]
    df_standards_followed = df[(df['standards_followed'] == 1)]
    df_standards_notfollowed = df[(df['standards_not_followed'] == 1)]

    print(f"Records with standards not mentioned: {len(df_standards_not_mentioned)}")
    print(f"Records with standards followed: {len(df_standards_followed)}")
    print(f"Records with standards not followed: {len(df_standards_notfollowed)}")

    if not config.include_disease_name:
        # Find all columns starting with "disease_"
        disease_columns = [col for col in df.columns if col.startswith('disease_')]
        df.drop(columns=disease_columns, inplace=True)

    # Focus on treatments and conditions if requested
    if config.analyze_treatments_conditions:
        # Retain treatment-related columns and relevant features
        treatment_columns = [col for col in df.columns if 'treatments_' in col]
        demographic_columns = [col for col in df.columns if any(x in col for x in ['gender_', 'age_range_'])]
        disease_columns = [col for col in df.columns if 'disease_' in col]
        
        columns_to_keep = (
            treatment_columns + 
            demographic_columns + 
            disease_columns + 
            ['is_denial_upheld']
        )
        
        df_standards_not_mentioned = df_standards_not_mentioned[columns_to_keep]
        df_standards_followed = df_standards_followed[columns_to_keep]
        df_standards_notfollowed = df_standards_notfollowed[columns_to_keep]
        df = df[columns_to_keep]

    # Run analysis for each subset
    if config.method == "random_forest":
        run_random_forest(df_standards_not_mentioned)
        run_random_forest(df_standards_followed)
        run_random_forest(df_standards_notfollowed)

    elif config.method == 'FPgrowth':
        run_FP(df_standards_not_mentioned)
        run_FP(df_standards_followed)
        run_FP(df_standards_notfollowed)

    elif config.method == "deap":
        base_output = config.output_filename()
        run_deap(df_standards_not_mentioned, f"results/nostandard_{base_output}")
        # run_deap(df_standards_followed, f"results/standards_followed_{base_output}")
        # run_deap(df_standards_notfollowed, f"results/standards_not_followed_{base_output}")
        run_deap(df, f"results/all_{base_output}")