from dataclasses import dataclass
from typing import Optional
import hashlib


@dataclass
class IMRQuery:
    """Class to hold query parameters for filtering IMR cases"""
    diagnosis_category: Optional[str] = ""
    diagnosis_subcategory: Optional[str] = ""
    determination: Optional[str] = ""
    treatment_category: Optional[str] = ""
    treatment_subcategory: Optional[str] = ""
    year: Optional[int] = None

    def name(self) -> str:
        """Construct a name based on available attributes."""
        parts = [self.diagnosis_category, self.diagnosis_subcategory, self.treatment_category, self.treatment_subcategory]
        return '-'.join(filter(None, parts))

@dataclass
class IMRConfig:
    """Configuration for IMR analysis including query parameters and analysis settings."""
    query: IMRQuery
    dataset_name: str
    method: str = "deap"  # Options: "random_forest", "FPgrowth", "deap"
    analyze_treatments_conditions: bool = False
    start_record: int = 0
    max_records: int = 37000
    input_csv: str = "data/ca-imr-determinations.csv"
    include_disease_name: bool = False

    def __post_init__(self):
        # Validate method
        valid_methods = ["random_forest", "FPgrowth", "deap"]
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    def output_filename(self) -> str:
        """Generate an output filename based on the query and method."""
        query_name = self.query.name().replace(" ", "_").lower()
        return f"{query_name}_{self.method}.csv"

    def generate_match_substring(self) -> str:
        """Generate the matching substring for file naming based on query hash and dataset name."""
        safe_desc = self.dataset_name.replace('/', '_').replace('\\', '_') # TODO tightly coupled to cache filename logic
        return f"{safe_desc}"