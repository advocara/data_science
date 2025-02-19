import json
import os
from typing import Dict, List, Optional, Set

import openai
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, Field, field_validator

from model.appeal import MedicalInsuranceAppeal
from util import get_openai_client, load_openai_key

class TermMappingSet(BaseModel):

    # Note: we actually want a dict here, but pydantic schema for a dict causes "required is required" error from OpenAI 
    # when returning a structured output. 
    mappinglist: List[List[str]] = Field(
        description="List of pairs mapping original terms to their normalized versions"
    )
    
    def add_all(self, original: List[str], normalized: str):
        for o in original:
            self.add(o, normalized)

    def add(self, original: str, normalized: str):
        """Add a mapping of a single original term to a single, normalized term"""
        # Check for existing term and replace if found
        for mapping in self.mappinglist:
            if mapping[0] == original:
                print(f"Warning: Replacing mapping for '{original}' from '{mapping[1]}' to '{normalized}'")
                mapping[1] = normalized
                return
        self.mappinglist.append([original, normalized]) # typical case: not found so append vs replace

    def get(self, original: str) -> str:
        """Get the normalized term for an original term, or the original term if it's not in the mapping"""
        for mapping in self.mappinglist:
            if original == mapping[0]:
                return mapping[1]  # Return normalized term
        return original  # Return original if not found
    
    def unique_keys(self) -> Set[str]:
        return set(mapping[0] for mapping in self.mappinglist)
    
    def unique_values(self) -> Set[str]:
        return set(mapping[1] for mapping in self.mappinglist)
    
# Add after TermMappingSet class definition
# print("Generated Schema:\n", TermMappingSet.model_json_schema())

class Normalizer:
        
    def __init__(self):
        self.client = get_openai_client()  # Use the utility function directly

    def get_normalized_mapping(self, items: Set[str], category: str) -> TermMappingSet: 
        """Get JSON structure that maps original terms to normalized terms to avoid duplicates due to name variations"""
        if not items:
            return TermMappingSet(mappinglist=[])

        prompt = f"""Given these medical {category}, return a JSON object that lists original terms with new, normalized terms.
        The mappinglist field should map different variations of terms to a single normalized version.
        
        Terms to normalize:
        {sorted(items)}
        """

        # Generate structured output using parse() with Pydantic model
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract the mappings."},
                {"role": "user", "content": prompt},
            ],
            response_format=TermMappingSet,
        )   

        # Parse the response JSON into our Pydantic model
        mappingset = completion.choices[0].message.parsed
        if mappingset is None:
            raise ValueError("No mappingset returned from LLM")
        print(f"Normalized: {mappingset.mappinglist}")
        return mappingset
        
    

    def normalize_names(self, appeals: List[MedicalInsuranceAppeal]) -> None:
        """Normalize names of treatments, conditions, and other fields using LLM to ensure consistency"""

        # First collect all unique names to normalize
        all_treatments = set()
        all_conditions = set()
        all_symptoms = set()
        all_complications = set()

        for appeal in appeals:
            # Collect treatments across all appeals
            for t in appeal.treatments_requested:
                all_treatments.add(t.name)
            for t in appeal.treatments_tried_but_failed:
                all_treatments.add(t.name)
            for t in appeal.treatments_tried_and_worked:
                all_treatments.add(t.name)
            for t in appeal.treatments_not_tried:
                all_treatments.add(t.name)
            
            # Collect other fields
            all_conditions.update(appeal.secondary_conditions)
            all_symptoms.update(appeal.symptoms)
            all_complications.update(appeal.complications)


        # Get mappings
        treatment_map = self.get_normalized_mapping(all_treatments, "treatments")
        condition_map = self.get_normalized_mapping(all_conditions, "conditions")
        symptom_map = self.get_normalized_mapping(all_symptoms, "symptoms")
        complication_map = self.get_normalized_mapping(all_complications, "complications")

        # Apply normalizations: replace the name of each treatment, condition, symptom, and complication with the normalized version
        for appeal in appeals:
            # First normalize all categories of treatments
            for t in appeal.treatments_requested:
                t.name = treatment_map.get(t.name)
            for t in appeal.treatments_tried_but_failed:
                t.name = treatment_map.get(t.name)
            for t in appeal.treatments_tried_and_worked:
                t.name = treatment_map.get(t.name)
            for t in appeal.treatments_not_tried:
                t.name = treatment_map.get(t.name)
            # now the other
            appeal.secondary_conditions = [condition_map.get(c) for c in appeal.secondary_conditions]
            appeal.symptoms = [symptom_map.get(s) for s in appeal.symptoms]
            appeal.complications = [complication_map.get(c) for c in appeal.complications]
    

def store_normalized_appeals(category_substring: str) -> None:
    """
    Read appeals from cache, normalize names, and store in appeals-results directory.
    
    Args:
        category_substring: String to match in cache filenames (e.g., 'Immuno Disorders-Lupus' for lupus appeals)
    """
    # Setup paths and create output directory
    cache_dir = './cache'
    results_dir = './appeals-results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if files already exist
    existing_files = [f for f in os.listdir(results_dir) 
                     if f.endswith('-norm.json') and category_substring in f]
    if existing_files:
        response = input(f"Found existing normalized files for '{category_substring}'. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping normalization...")
            return
    
    # Get matching cache files
    cache_files = [f for f in os.listdir(cache_dir) 
                  if f.endswith('.json') and category_substring in f]
    
    if not cache_files:
        print(f"No cache files found matching '{category_substring}'")
        return
        
    # Load all appeals
    appeals = []
    for filename in cache_files:
        with open(os.path.join(cache_dir, filename), 'r', encoding='utf-8') as f:
            appeal_data = json.load(f)
            appeal = MedicalInsuranceAppeal.model_validate(appeal_data)
            appeals.append(appeal)
    
    print(f"Read in {len(appeals)} appeals. Normalizing names/features...")
    normalizer = Normalizer()
    normalizer.normalize_names(appeals)
    
    # Write normalized versions
    print(f"Writing normalized versions to {results_dir}...")
    for filename in cache_files:
        base_name = filename.replace('.json', '')
        norm_filename = f"{base_name}-norm.json"
        appeal_index = cache_files.index(filename)
        
        with open(os.path.join(results_dir, norm_filename), 'w', encoding='utf-8') as f:
            json.dump(appeals[appeal_index].model_dump(), f, indent=2)
    
    print(f"Processed {len(appeals)} appeals and saved normalized versions to {results_dir}")

if __name__ == "__main__":
    print("=======") 
    print(TermMappingSet.model_json_schema())
    print("=======") 


