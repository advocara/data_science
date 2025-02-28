import unittest
from typing import List, Set
from model.appeal import MedicalInsuranceAppeal, Treatment
from term_normalizer import Normalizer

class TestNormalizer(unittest.TestCase):
    test_appeals = None  
    
    def setUp(self):
        self.normalizer = Normalizer()
        if not TestNormalizer.test_appeals:  
            TestNormalizer.test_appeals = self.load_test_appeals()

    def test_normalization(self):
    
        non_normalized = set(["physical therapy", "physical therapist support", "ibuprofen", "ibuprofen 800 mg", "advil"])
        normalized = self.normalizer.get_normalized_mapping(
            non_normalized,
              "treatments")
        print(f"Normalized: {normalized}")

        self.assertGreater(len(normalized.unique_keys()), len(normalized.unique_values()), 
                        f"No collapse of similar terms done \n{normalized.unique_keys()} \nvs {normalized.unique_values()}")
        self.assertIsNotNone(normalized.get("physical therapy"))
        self.assertIsNotNone(normalized.get("physical therapy support"))
        self.assertIsNotNone(normalized.get("ibuprofen"), normalized.get("ibuprofen 800 mg"))
        self.assertIsNotNone(normalized.get("physical therapy"), normalized.get("physical therapist support"))
        self.assertIsNotNone(normalized.get("advil"))
        self.assertIsNotNone(normalized.get("physical therapist support"))


    def load_test_appeals(self) -> List[MedicalInsuranceAppeal]:
        test_appeals = []
        test_files = [
            "testappeal_lupus_1.json",
            "testappeal_lupus_2.json", 
            "testappeal_lupus_3.json",
            "testappeal_lupus_4.json",
            "testappeal_lupus_5.json"
        ]
        
        for filename in test_files:
            with open(f"test_data/{filename}", "r") as f:
                appeal = MedicalInsuranceAppeal.model_validate_json(f.read())
            test_appeals.append(appeal)
            
        return test_appeals

    def _get_matching_treatment_names(self, find_strings: List[str], appeals: List[MedicalInsuranceAppeal]) -> Set[str]:
        matching_names = set()
        for appeal in appeals:
            for treatment in appeal.treatments_requested + appeal.treatments_tried_and_worked + appeal.treatments_tried_but_failed:
                if any(find_string in treatment.name.lower() for find_string in find_strings):
                    matching_names.add(treatment.name)
        return matching_names
    
    def _get_matching_conditions(self, find_string: str, appeals: List[MedicalInsuranceAppeal]) -> Set[str]:
        matching_conditions = set()
        for appeal in appeals:
            for condition in appeal.secondary_conditions:
                if find_string in condition.lower():
                    matching_conditions.add(condition)
        return matching_conditions
    
    def _get_matching_symptoms(self, find_string: str, appeals: List[MedicalInsuranceAppeal]) -> Set[str]:
        matching_symptoms = set()
        for appeal in appeals:
            for symptom in appeal.symptoms:
                if find_string in symptom.lower():
                    matching_symptoms.add(symptom)
        return matching_symptoms
    
    def test_normalize_names(self):
        
        assert TestNormalizer.test_appeals is not None
        appeals: List[MedicalInsuranceAppeal] = TestNormalizer.test_appeals

        # Find all Wegovy treatments before normalization
        wegovy_names_orig = self._get_matching_treatment_names(["wegovy" 'semaglutide'], appeals)
        lupus_conditions_orig = self._get_matching_conditions("lupus", appeals)

        # Normalize the appeals
        self.normalizer.normalize_names(appeals)

        # Check that variations were normalized
        # Find all Wegovy treatments after normalization
        wegovy_names_norm = self._get_matching_treatment_names(["wegovy", "semaglutide"], appeals)
        
        ### check treatments ###
        # Check that all Wegovy treatment names were normalized to the same string
        self.assertEqual(
            len(wegovy_names_norm), 
            1,
            f"Wegovy names were not normalized to a single format: {wegovy_names_norm}"
        )


if __name__ == '__main__':
    unittest.main()
