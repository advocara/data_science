import os
from typing import List, Dict
from cache import FileCache
from imr_analyzer import IMRAnalyzer, IMRQuery
from results.model.appeal import MedicalInsuranceAppeal
from util import load_openai_key

def create_theme_instructions() -> str:
    """Create focused instructions for actionable theme extraction"""
    return """Please identify specific, actionable themes in these medical review cases, focusing on:
    1. Denial Patterns:
       - Specific reasons given for denial
       - Common documentation gaps
       - Policy interpretation issues
    
    2. Successful Appeals:
       - Key factors in overturned decisions
       - Effective documentation strategies
       - Compelling medical necessity arguments
    
    3. Process Requirements:
       - Critical documentation needed
       - Timing considerations
       - Prior authorization requirements
       - Required pre-approvals or tests
    
    4. Treatment Access:
       - Commonly approved/denied treatments
       - Alternative treatment requirements
       - Step therapy patterns
       
    5. Coverage Challenges:
       - Off-label medication issues
       - Experimental treatment considerations
       - Insurance policy limitations
    
    For each theme, include:
    - Specific examples that illustrate the pattern
    - Frequency of occurrence
    - Notable successful strategies
    """

def main():
    # Configure parameters
    input_csv = "data/ca-imr-determinations.csv"
    start_record = 0  # Start from first record
    max_records = 5  # Process lmit of records
    chunk_size = 4    # Process in chunks of 4
    
    # Load OpenAI API key
    try:
        api_key = load_openai_key()
        print("Successfully loaded OpenAI API key")
    except FileNotFoundError:
        print("Error: Please create ~/openai.key file with your OpenAI API key")
        return
    
    # Initialize analyzer
    analyzer = IMRAnalyzer(input_csv, api_key)
    
    # Create query for Lupus cases
    query = IMRQuery(
        diagnosis_category="Immuno Disorders",
        diagnosis_subcategory="Lupus"
    )
    dataset_name = f'{query.diagnosis_category}-{query.diagnosis_subcategory}'

    # Find matching cases
    matches = analyzer.find_matching_cases(query, start_record=start_record, max_records=max_records)
    total_found = len(analyzer.find_matching_cases(query))  # Get total without limits
    print(f"\nFound {total_found} total cases matching Lupus under Immuno Disorders")
    print(f"Processing {len(matches)} cases starting from record {start_record}")
    
    if len(matches) == 0:
        print("No matching cases found")
        return
    
    # Analyze themes in each IMR
    results: List[MedicalInsuranceAppeal] = []
    for match in matches:
        appeal = analyzer.extract_appeal(match, create_theme_instructions(), dataset_name)
        results.append(appeal)



if __name__ == "__main__":
    main()
