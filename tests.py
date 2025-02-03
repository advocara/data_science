import unittest
from typing import List
from results.model.appeal import MedicalInsuranceAppeal, Treatment
from term_normalizer import Normalizer

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = Normalizer()

    def test_normalize_names(self):
        # Create test appeals with variations in naming
        treatment_requested = Treatment(name="Physical Therapy", is_procedure=True, drug_type=None, procedure_type=None, other_or_notes=None)
        treatment_tried_but_failed = Treatment(name="PT", is_procedure=True, drug_type=None, procedure_type=None, other_or_notes=None)
        treatment_tried_and_worked = Treatment(name="physical therapy", is_procedure=True, drug_type=None, procedure_type=None, other_or_notes=None)
        treatment_not_tried = Treatment(name="PT", is_procedure=True, drug_type=None, procedure_type=None, other_or_notes=None)
        appeals: List[MedicalInsuranceAppeal] = [
            MedicalInsuranceAppeal(
                case_id="test1",
                year=2023,
                diagnosis="test diagnosis",
                determination="DENIAL_UPHELD",
                expedited=False,
                guidelines_support=True,
                soc_support=True,
                study_support=True,
                treatment_category="test category",
                treatment_subcategory="test subcategory",
                treatments_requested=[treatment_requested],
                treatments_tried_but_failed=[treatment_tried_but_failed],
                treatments_tried_and_worked=[treatment_tried_and_worked],
                treatments_not_tried=[],
                secondary_conditions=["Diabetes Type 2", "Type II Diabetes"],
                symptoms=["back pain", "Back Pain"],
                complications=["infection", "Infection"],
                issues_considered=[],
                key_questions=[],
                study_details=[],
                other_issues=None,
                guidelines_not_support=False,
                guidelines_details=None,
                soc_not_support=False,
                soc_details=None,
                rationale=None,
                reviewer_credentials=None
            )
        ]

        # Normalize the appeals
        normalized = self.normalizer.normalize_names(appeals)

        # Check that variations were normalized
        appeal = normalized[0]
        
        # Check treatments were normalized to consistent naming
        self.assertEqual(appeal.treatments_requested[0].name, 
                        appeal.treatments_tried_and_worked[0].name)
        self.assertEqual(appeal.treatments_requested[0].name,
                        appeal.treatments_tried_but_failed[0].name)

        # Check conditions were normalized
        self.assertEqual(len(set(appeal.secondary_conditions)), 1)

        # Check symptoms were normalized
        self.assertEqual(len(set(appeal.symptoms)), 1)

        # Check complications were normalized
        self.assertEqual(len(set(appeal.complications)), 1)

if __name__ == '__main__':
    unittest.main()
