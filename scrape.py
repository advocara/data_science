import os
from typing import List, Dict
from cache import FileCache
from imr_analyzer import IMRAnalyzer, IMRQuery
from model.appeal import MedicalInsuranceAppeal
from model.config import IMRConfig
from util import load_openai_key

def gen_cache(input_csv, config: IMRConfig):
    
    print(f"======= \n======= Processing {config.max_records} cases starting from record {config.start_record} \n=======")

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

    dataset_name = config.query.name()

    # Find matching cases
    matches = analyzer.find_matching_cases(config.query, start_record=config.start_record, max_records=config.max_records)
    total_found = len(analyzer.find_matching_cases(config.query))  # Get total without limits
    print(f"\nFound {total_found} total cases matching {dataset_name}")
    print(f"Processing {len(matches)} cases starting from record {config.start_record}")
    
    if len(matches) == 0:
        print("No matching cases found")
        return
    
    # Analyze themes in each IMR
    results: List[MedicalInsuranceAppeal] = []
    for i, match in enumerate(matches):
        record_id = match.reference_id
        appeal = analyzer.extract_appeal(match, "Extract data to match the schema", config, record_id)
        results.append(appeal)
        print(f"   above was ({i+1}/{len(matches)})")
        # results ignored because the process stores them all in the /cache directory
