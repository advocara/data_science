from pydantic import BaseModel, Field
from typing import List, Literal, Optional


IssueType = Literal["medical_necessity", "diagnosis", "plan_coverage", "preventative_care_rules", "cost_considerations", "other"]
ProcedureType = Literal["surgery", "imaging", "physical_therapy", "radiation", "psychotherapy", "diagnostic", "rehabilitation", "preventive", "other", "unspecified"]
DrugType = Literal["biologic", "infusion", "experimental", "standard", "other", "unspecified"]
KeyQuestionType = Literal["medical_necessity", "confirm_diagnosis", "cost_justification", "plan_coverage", "over_limits", "other"]
GenderType = Literal["Male", "Female", "Other", "Unspecified"]


class IMRRow:
    def __init__(
        self,
        reference_id: str,
        report_year: int,
        diagnosis_category: str,
        diagnosis_sub_category: str,
        treatment_category: str,
        treatment_sub_category: str,
        determination: str,
        type: str,
        age_range: str,
        patient_gender: str,
        imr_type: str,
        days_to_review: int,
        days_to_adopt: int,
        findings: str
    ):
        self.reference_id = reference_id
        self.report_year = report_year
        self.diagnosis_category = diagnosis_category
        self.diagnosis_sub_category = diagnosis_sub_category
        self.treatment_category = treatment_category
        self.treatment_sub_category = treatment_sub_category
        self.determination = determination
        self.type = type
        self.age_range = age_range
        self.patient_gender = patient_gender
        self.imr_type = imr_type
        self.days_to_review = days_to_review
        self.days_to_adopt = days_to_adopt
        self.findings = findings

    def to_string(self): 
        # hack. we know this is inside a 8-char indented prompt, so indent 12.
        return f"""
            Case ID: {self.reference_id}
            Report Year: {self.report_year}
            Diagnosis Category: {self.diagnosis_category}
            Diagnosis Subcategory: {self.diagnosis_sub_category}
            Treatment Category: {self.treatment_category}
            Treatment Subcategory: {self.treatment_sub_category}
            Determination: {self.determination}
            Type: {self.type}
            Age Range: {self.age_range}
            Patient Gender: {self.patient_gender}
            IMR Type: {self.imr_type}
            Days to Review: {self.days_to_review}
            Days to Adopt: {self.days_to_adopt}
            Findings: {self.findings}"""

class PatientInfo(BaseModel):
    age_range: Optional[str] = Field(
        None, 
        description="Age range of the patient."
    )
    gender: Optional[GenderType] = Field(
        None, 
        description="Gender of the patient."
    )

class Treatment(BaseModel):
    name: str = Field(..., description="Name of the treatment (drug or procedure)")
    drug_type: Optional[DrugType] = Field(
        None, 
        description="Type of drug"
    )
    procedure_type: Optional[ProcedureType] = Field(
        None, 
        description="Type of procedure"
    )
    is_procedure: bool = Field(
        description="Indicates if this is a procedure rather than a drug"
    )
    other_or_notes: Optional[str] = Field(
        None, 
        description="Additional details about the treatment, especially if type is 'other'"
    )


class StudyDetails(BaseModel):
    study_name: str = Field(description="Name of the study referenced.")
    study_authors: str = Field(description="Authors of the referenced study.")
    key_findings: str = Field(description="Summary of the study findings.")


class MedicalInsuranceAppeal(BaseModel):
    case_id: str = Field(..., description="Unique identifier for the appeal case.")
    year: int = Field(..., description="Year of the appeal decision.")

    patient_info: Optional[PatientInfo] = None

    diagnosis: str = Field(..., description="Primary medical condition related to the appeal.")
    secondary_conditions: List[str] = Field(
        default_factory=list, description="Secondary conditions or comorbidities associated with the appeal."
    )
    complications: List[str] = Field(
        default_factory=list, description="Complications that impact medical necessity or treatment effectiveness."
    )
    symptoms: List[str] = Field(
        default_factory=list, description="List of symptoms relevant to the appeal."
    )

    treatment_category: Optional[str] = Field(None, description="General category of the treatment requested.")
    treatment_subcategory: Optional[str] = Field(None, description="More specific classification of the treatment.")

    treatments_requested: List[Treatment] = Field(
        default_factory=list,
        description="Treatments being requested in this appeal"
    )
    treatments_tried_but_failed: List[Treatment] = Field(
        default_factory=list,
        description="Treatments that were attempted but did not provide adequate results"
    )
    treatments_tried_and_worked: List[Treatment] = Field(
        default_factory=list,
        description="Treatments that were attempted and showed positive results"
    )
    treatments_not_tried: List[Treatment] = Field(
        default_factory=list,
        description="Relevant treatments that have not been attempted, but are indicated for the condition or otherwise relevant."
    )

    is_denial_upheld: bool = Field(
        description="Indicates if the insurance company denial decision was upheld."
    )

    issues_considered: List[IssueType] = Field(
        default_factory=list,
        description="Key issues that were considered in making the appeal decision."
    )
    other_issues: Optional[str] = Field(
        None, 
        description="Additional issues that were considered in making the appeal decision, if issues_considered is 'other'."
    )
    
    guidelines_support: Optional[bool] = Field(
        None, description="Indicates if formal guidelines support the requested treatment."
    )
    guidelines_not_support: Optional[bool] = Field(
        None, description="Indicates if formal guidelines do not support the requested treatment."
    )
    guidelines_details: Optional[str] = Field(
        None, description="Summary of relevant guidelines and their position on the requested treatment."
    )
    soc_support: Optional[bool] = Field(
        None, description="Indicates if standards of care support the requested treatment."
    )
    soc_not_support: Optional[bool] = Field(
        None, description="Indicates if standards of care do not support the requested treatment."
    )
    soc_details: Optional[str] = Field(
        None, description="Summary of relevant standards of care and their position on the requested treatment."
    )

    study_support: Optional[bool] = Field(
        None, description="Indicates if a study was referenced to support the requested treatment."
    )
    study_details: List[StudyDetails] = Field(
        default_factory=list, description="Details of studies that were referenced in the appeal."
    )

    key_questions: List[KeyQuestionType] = Field(
        default_factory=list,
        description="Key questions that arose in the appeal decision process."
    )

    expedited: Optional[bool] = Field(
        None, description="Indicates if the appeal was processed as an expedited request."
    )

    rationale: Optional[str] = Field(None, description="Summary of the reasoning behind the appeal decision.")
    reviewer_credentials: Optional[str] = Field(None, description="Details about the reviewer's medical expertise.")

    class Config:
        populate_by_name = True
