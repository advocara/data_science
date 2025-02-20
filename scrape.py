import os
from typing import List, Dict
from cache import FileCache
from imr_analyzer import IMRAnalyzer, IMRQuery
from model.appeal import MedicalInsuranceAppeal
from util import load_openai_key


def gen_cache(input_csv, query: IMRQuery, start_record = 0, max_records = 37000):
    
    print(f"======= \n======= Processing {max_records} cases starting from record {start_record} \n=======")

    # Load OpenAI API key
    try:
        api_key = load_openai_key()
        print("Successfully loaded OpenAI API key")
    except FileNotFoundError:
        print(os.path.expanduser('~'))
        print("Error: Please create ~/openai.key file with your OpenAI API key")
        return
    
    # Initialize analyzer
    analyzer = IMRAnalyzer(input_csv, api_key)

    dataset_name = f'{query.diagnosis_category}-{query.diagnosis_subcategory}'

    # Find matching cases
    matches = analyzer.find_matching_cases(query, start_record=start_record, max_records=max_records)
    total_found = len(analyzer.find_matching_cases(query))  # Get total without limits
    print(f"\nFound {total_found} total cases matching {dataset_name}")
    print(f"Processing {len(matches)} cases starting from record {start_record}")
    
    if len(matches) == 0:
        print("No matching cases found")
        return
    
    # Analyze themes in each IMR
    results: List[MedicalInsuranceAppeal] = []
    for i, match in enumerate(matches):
        appeal = analyzer.extract_appeal(match, "Extract data to match the schema", dataset_name)
        results.append(appeal)
        print(f"   above was ({i+1}/{len(matches)})")
