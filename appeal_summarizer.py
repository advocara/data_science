"""

"""
from typing import List

from model.appeal import MedicalInsuranceAppeal

import pandas as pd

def flatten_appeals_to_dataframe_OBSOLETE(appeals: List[MedicalInsuranceAppeal]) -> pd.DataFrame:
    # First collect all unique values for one-hot encoding
    unique_values = {
        'secondary_conditions': set(),
        'complications': set(),
        'symptoms': set(),
        'treatments_requested': set(),
        'treatments_tried_but_failed': set(),
        'treatments_tried_and_worked': set(),
        'treatments_not_tried': set(),
        'issues_considered': set(),
        'key_questions': set()
    }
    
    # Collect all unique values
    for appeal in appeals:
        unique_values['secondary_conditions'].update(appeal.secondary_conditions)
        unique_values['complications'].update(appeal.complications)
        unique_values['symptoms'].update(appeal.symptoms)
        
        # Handle treatments - collect names for each treatment category
        for treatment in appeal.treatments_requested:
            unique_values['treatments_requested'].add(treatment.name)
        for treatment in appeal.treatments_tried_but_failed:
            unique_values['treatments_tried_but_failed'].add(treatment.name)
        for treatment in appeal.treatments_tried_and_worked:
            unique_values['treatments_tried_and_worked'].add(treatment.name)
        for treatment in appeal.treatments_not_tried:
            unique_values['treatments_not_tried'].add(treatment.name)
            
        unique_values['issues_considered'].update(appeal.issues_considered)
        unique_values['key_questions'].update(appeal.key_questions)

    # Create rows for DataFrame
    rows = []
    for appeal in appeals:
        row = {
            'case_id': appeal.case_id,
            'year': appeal.year,
            'diagnosis': appeal.diagnosis,
            'determination': appeal.determination,
            'treatment_category': appeal.treatment_category,
            'treatment_subcategory': appeal.treatment_subcategory,
            'expedited': appeal.expedited,
            'guidelines_support': appeal.guidelines_support,
            'guidelines_not_support': appeal.guidelines_not_support,
            'soc_support': appeal.soc_support,
            'soc_not_support': appeal.soc_not_support,
            'study_support': appeal.study_support
        }
        
        # Add patient info if exists
        if appeal.patient_info:
            row['patient_age_range'] = appeal.patient_info.age_range
            row['patient_gender'] = appeal.patient_info.gender

        # Add one-hot encoded columns for lists
        for condition in unique_values['secondary_conditions']:
            row[f'secondary_condition_{condition}'] = condition in appeal.secondary_conditions
            
        for complication in unique_values['complications']:
            row[f'complication_{complication}'] = complication in appeal.complications
            
        for symptom in unique_values['symptoms']:
            row[f'symptom_{symptom}'] = symptom in appeal.symptoms
            
        # Handle treatments
        for treatment in unique_values['treatments_requested']:
            row[f'requested_{treatment}'] = any(t.name == treatment for t in appeal.treatments_requested)
            
        for treatment in unique_values['treatments_tried_but_failed']:
            row[f'failed_{treatment}'] = any(t.name == treatment for t in appeal.treatments_tried_but_failed)
            
        for treatment in unique_values['treatments_tried_and_worked']:
            row[f'worked_{treatment}'] = any(t.name == treatment for t in appeal.treatments_tried_and_worked)
            
        for treatment in unique_values['treatments_not_tried']:
            row[f'not_tried_{treatment}'] = any(t.name == treatment for t in appeal.treatments_not_tried)
            
        for issue in unique_values['issues_considered']:
            row[f'issue_{issue}'] = issue in appeal.issues_considered
            
        for question in unique_values['key_questions']:
            row[f'question_{question}'] = question in appeal.key_questions

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert determination to binary for easier analysis
    df['determination_overturned'] = (df['determination'] == 'DENIAL_OVERTURNED').astype(int)
    
    return df

def print_column_summary(df: pd.DataFrame):
    """Print a summary of the columns in the DataFrame by category"""
    print("Base columns:", [col for col in df.columns if '_' not in col])
    print("\nSecondary conditions:", [col for col in df.columns if col.startswith('secondary_condition_')])
    print("\nComplications:", [col for col in df.columns if col.startswith('complication_')])
    print("\nSymptoms:", [col for col in df.columns if col.startswith('symptom_')])
    print("\nTreatments requested:", [col for col in df.columns if col.startswith('requested_')])
    print("\nTreatments failed:", [col for col in df.columns if col.startswith('failed_')])
    print("\nTreatments worked:", [col for col in df.columns if col.startswith('worked_')])
    print("\nTreatments not tried:", [col for col in df.columns if col.startswith('not_tried_')])
    print("\nIssues:", [col for col in df.columns if col.startswith('issue_')])
    print("\nQuestions:", [col for col in df.columns if col.startswith('question_')])

