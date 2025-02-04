from typing import List, Dict, Optional, Set, Union, TypedDict
from dataclasses import dataclass
from collections import Counter
import os
import json

import pandas as pd  # type: ignore
import openai
from openai import OpenAI

from cache import FileCache  # Update import
from results.model.appeal import IMRRow, MedicalInsuranceAppeal
from util import get_openai_client, load_openai_key

MAX_CASES = 20
CHUNK_SIZE = 4

@dataclass
class IMRQuery:
    """Class to hold query parameters for filtering IMR cases"""
    diagnosis_category: Optional[str] = None
    diagnosis_subcategory: Optional[str] = None
    determination: Optional[str] = None
    treatment_category: Optional[str] = None
    treatment_subcategory: Optional[str] = None
    year: Optional[int] = None

class IMRAnalyzer:
    def __init__(self, csv_path: str, openai_api_key: Optional[str]):
        """Initialize the IMR analyzer with data path and OpenAI credentials"""
        self.df = pd.read_csv(csv_path)
        self.client = get_openai_client()
        self.cache = FileCache()  # Initialize the cache
        
    def find_matching_cases(self, query: IMRQuery, start_record: int = 0, max_records: int = 100) -> List[IMRRow]:
        """Filter cases based on query parameters and return IMRRow objects"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if query.diagnosis_category:
            filtered_df = filtered_df[filtered_df['DiagnosisCategory'] == query.diagnosis_category]
        if query.diagnosis_subcategory:
            filtered_df = filtered_df[filtered_df['DiagnosisSubCategory'] == query.diagnosis_subcategory]
        if query.determination:
            filtered_df = filtered_df[filtered_df['Determination'] == query.determination]
        if query.treatment_category:
            filtered_df = filtered_df[filtered_df['TreatmentCategory'] == query.treatment_category]
        if query.treatment_subcategory:
            filtered_df = filtered_df[filtered_df['TreatmentSubCategory'] == query.treatment_subcategory]
        if query.year:
            filtered_df = filtered_df[filtered_df['ReportYear'] == query.year]

        # Apply pagination
        filtered_df = filtered_df.iloc[start_record:start_record + max_records]
        
        # Convert each row to IMRRow
        imr_rows = []
        for _, row in filtered_df.iterrows():
            imr_row = IMRRow(
                reference_id=row['ReferenceID'],
                report_year=row['ReportYear'],
                diagnosis_category=row['DiagnosisCategory'],
                diagnosis_sub_category=row['DiagnosisSubCategory'],
                treatment_category=row['TreatmentCategory'],
                treatment_sub_category=row['TreatmentSubCategory'],
                determination=row['Determination'],
                type=row['Type'],
                age_range=row['AgeRange'],
                patient_gender=row['PatientGender'],
                imr_type=row['IMRType'],
                days_to_review=row['DaysToReview'],
                days_to_adopt=row['DaysToAdopt'],
                findings=row['Findings']
            )
            imr_rows.append(imr_row)
        
        return imr_rows

    def extract_appeal(self, imr: IMRRow, instructions: str, dataset_name: str) -> MedicalInsuranceAppeal:
        """Extract themes from a group of cases using OpenAI's GPT-4"""
        # Check cache first
        full_text = imr.to_string()
        cached_result = self.cache.get(full_text, dataset_name)
        if cached_result:
            try:
                print(f"Cache hit for {full_text[:200]}")
                return json.loads(cached_result)
            except json.JSONDecodeError:
                print("Cache format is invalid. Clear or fix incompatible cache JSON and re-run")
                raise

        prompt = f"""Analyze the following medical review case and return a formatted object with summarized text:
        
        {instructions}
        
        Case:
        {imr.to_string()}
        """
        client = openai.OpenAI(api_key=load_openai_key())
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=MedicalInsuranceAppeal,
            temperature=0.0,
        )
        
        mia = response.choices[0].message.parsed
        if mia is None:
            raise ValueError("No object returned from OpenAI")
            
        assert isinstance(mia, MedicalInsuranceAppeal)
        # Cache the successful result as the object JSON form
        self.cache.put(full_text, mia.model_dump_json(), dataset_name)

        print(f"Extracted and cached case {mia.case_id}")

        return mia
            # Create normalization mappings using LLM
    
    
    def appeals_to_df(self, appeals: List[MedicalInsuranceAppeal]) -> pd.DataFrame:
        """Convert MedicalInsuranceAppeals into rows in a data frame
        using one-hot encoding for categorical columns and lists"""

        # Convert appeals to DataFrame for analysis
        appeals_data = []
        for appeal in appeals:
            appeal_dict = {
                'case_id': appeal.case_id,
                'year': appeal.year,
                'diagnosis': appeal.diagnosis,
                'determination': appeal.determination,
                'expedited': appeal.expedited,
                'guidelines_support': appeal.guidelines_support,
                'soc_support': appeal.soc_support,
                'study_support': appeal.study_support,
                'age_range': appeal.patient_info.age_range if appeal.patient_info else None,
                'gender': appeal.patient_info.gender if appeal.patient_info else None,
                'treatment_category': appeal.treatment_category,
                'treatment_subcategory': appeal.treatment_subcategory,
            }

            # One-hot encode secondary conditions
            for condition in appeal.secondary_conditions:
                appeal_dict[f'condition_{condition.lower().replace(" ", "_")}'] = 1

            # One-hot encode complications
            for complication in appeal.complications:
                appeal_dict[f'complication_{complication.lower().replace(" ", "_")}'] = 1

            # One-hot encode symptoms
            for symptom in appeal.symptoms:
                appeal_dict[f'symptom_{symptom.lower().replace(" ", "_")}'] = 1

            # One-hot encode treatments requested
            for treatment in appeal.treatments_requested:
                appeal_dict[f'requested_{treatment.name.lower().replace(" ", "_")}'] = 1

            # One-hot encode treatments tried but failed
            for treatment in appeal.treatments_tried_but_failed:
                appeal_dict[f'failed_{treatment.name.lower().replace(" ", "_")}'] = 1

            # One-hot encode treatments that worked
            for treatment in appeal.treatments_tried_and_worked:
                appeal_dict[f'worked_{treatment.name.lower().replace(" ", "_")}'] = 1

            # One-hot encode treatments not tried
            for treatment in appeal.treatments_not_tried:
                appeal_dict[f'nottried_{treatment.name.lower().replace(" ", "_")}'] = 1

            # One-hot encode issues considered
            for issue in appeal.issues_considered:
                appeal_dict[f'issue_{issue.lower()}'] = 1

            # One-hot encode key questions
            for question in appeal.key_questions:
                appeal_dict[f'question_{question.lower()}'] = 1

            # Add counts as well
            appeal_dict.update({
                'num_treatments_requested': len(appeal.treatments_requested),
                'num_treatments_failed': len(appeal.treatments_tried_but_failed),
                'num_treatments_worked': len(appeal.treatments_tried_and_worked),
                'num_treatments_not_tried': len(appeal.treatments_not_tried),
                'num_secondary_conditions': len(appeal.secondary_conditions),
                'num_complications': len(appeal.complications),
                'num_symptoms': len(appeal.symptoms),
                'num_issues': len(appeal.issues_considered),
                'num_studies': len(appeal.study_details)
            })

            appeals_data.append(appeal_dict)
        
        # Create DataFrame and fill NaN values with 0 for one-hot encoded columns
        df = pd.DataFrame(appeals_data)
        
        # Fill NaN values with 0 for boolean columns (one-hot encoded)
        bool_columns = [col for col in df.columns if any(prefix in col for prefix in 
                       ['condition_', 'complication_', 'symptom_', 'requested_', 
                        'failed_', 'worked_', 'nottried_', 'issue_', 'question_'])]
        df[bool_columns] = df[bool_columns].fillna(0)
        
        return df

    def print_records_to_markdown(self, query: IMRQuery, n: int = 5) -> None:
        """Print N records matching the query criteria to a markdown file
           Useful for human review of cases."""
        # Get matching cases
        cases = self.find_matching_cases(query, max_records=n)
        
        # Create filename
        diag = query.diagnosis_category or 'all'
        subcat = query.diagnosis_subcategory or 'all'
        filename = f"view-{n}-{diag}-{subcat}.md".replace(' ', '_').lower()
        
        # Create markdown content
        content = f"# IMR Cases Review: {diag} - {subcat}\n\n"
        content += f"Showing {len(cases)} cases\n\n"
        
        # Add each case
        for i, case in enumerate(cases, 1):
            content += f"## Case {i}\n\n"
            content += f"**Year:** {case.report_year}\n\n"
            content += f"**Diagnosis:** {case.diagnosis_category} - {case.diagnosis_sub_category}\n\n"
            content += f"**Treatment:** {case.treatment_category} - {case.treatment_sub_category}\n\n"
            content += f"**Determination:** {case.determination}\n\n"
            content += "**Findings:**\n\n"
            content += f"{case.findings}\n\n"
            content += "---\n\n"
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
