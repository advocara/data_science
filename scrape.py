import os
from typing import List, Dict
from cache import FileCache
from imr_analyzer import IMRAnalyzer, IMRQuery
from model.appeal import MedicalInsuranceAppeal
from util import load_openai_key


def main():
    # Configure parameters
    input_csv = "data\ca-imr-determinations.csv"
    start_record = 0  # Start from first record
    max_records = 37000  # Process lmit of records
    chunk_size = 4    # Process in chunks of 4
    
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
    
    # Create query for Lupus cases
    query = IMRQuery(
        # diagnosis_category="Immuno Disorders",
        # diagnosis_subcategory="Lupus"
        # diagnosis_category="Alzheimer's Disease",
        diagnosis_subcategory="Fibromyalgia"
    )
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



if __name__ == "__main__":
    main()
