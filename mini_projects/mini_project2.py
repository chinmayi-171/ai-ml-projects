# ==============================
# Project 2: AI Medical Report Extractor
# ==============================

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
from typing import List


# ==============================
# Load LLM
# ==============================

llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)


# ============================================================
# FEATURE 1: Patient Information Extraction
# ============================================================

class PatientInfo(BaseModel):
    patient_name: str
    age: int
    gender: str
    diagnosis: str
    prescribed_medications: List[str]


patient_parser = PydanticOutputParser(pydantic_object=PatientInfo)

patient_prompt = PromptTemplate(
    template="""
Extract structured patient information from the medical report.

Follow the format instructions strictly.
Do not add extra text.

{format_instructions}

Medical Report:
{report}
""",
    input_variables=["report"],
    partial_variables={
        "format_instructions": patient_parser.get_format_instructions()
    }
)

patient_chain = patient_prompt | llm | patient_parser


# ============================================================
# FEATURE 2: Risk Assessment
# ============================================================

class RiskAssessment(BaseModel):
    severity_level: str
    critical_findings: List[str]
    recommended_actions: List[str]


risk_parser = PydanticOutputParser(pydantic_object=RiskAssessment)

risk_prompt = PromptTemplate(
    template="""
Analyze the medical report and generate a structured risk assessment.

Rules:
- Severity must be one of: Low, Moderate, High, Critical.
- Follow the format instructions strictly.

{format_instructions}

Medical Report:
{report}
""",
    input_variables=["report"],
    partial_variables={
        "format_instructions": risk_parser.get_format_instructions()
    }
)

risk_chain = risk_prompt | llm | risk_parser


# ============================================================
# FEATURE 3: Doctor-Friendly Summary
# ============================================================

summary_prompt = PromptTemplate(
    template="""
Generate a concise and professional medical summary suitable for a doctor.

Medical Report:
{report}
""",
    input_variables=["report"]
)

summary_chain = summary_prompt | llm | StrOutputParser()


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_medical_extractor(report_text):

    # Feature 1
    patient_info = patient_chain.invoke({"report": report_text})

    print("\n===== Patient Information Extraction =====")
    print(f"\nPatient Name: {patient_info.patient_name}")
    print(f"Age: {patient_info.age}")
    print(f"Gender: {patient_info.gender}")
    print(f"Diagnosis: {patient_info.diagnosis}")

    print("\nPrescribed Medications:")
    for med in patient_info.prescribed_medications:
        print(f"- {med}")

    # Feature 2
    risk_info = risk_chain.invoke({"report": report_text})

    print("\n===== Risk Assessment Summary =====")
    print(f"\nSeverity Level: {risk_info.severity_level}")

    print("\nCritical Findings:")
    for finding in risk_info.critical_findings:
        print(f"- {finding}")

    print("\nRecommended Actions:")
    for action in risk_info.recommended_actions:
        print(f"- {action}")

    # Feature 3
    summary = summary_chain.invoke({"report": report_text})

    print("\n===== Doctor-Friendly Summary =====\n")
    print(summary)


# ============================================================
# USER INPUT SECTION
# ============================================================

if __name__ == "__main__":

    print("===================================")
    print("   AI Medical Report Extractor")
    print("===================================")

    print("\nEnter Patient Information (Paste paragraph and press ENTER once to finish):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    medical_report = "\n".join(lines).strip()

    run_medical_extractor(medical_report)