# ==============================
# AI Job Application Assistant
# Using Ollama (phi3:mini)
# ==============================

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
from typing import List


# ==============================
# Load LLM (phi3 mini)
# ==============================

llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)


# ==============================
# FEATURE 1: Job Description Analyzer
# ==============================

class JobDetails(BaseModel):
    job_title: str = Field(description="Title of the job")
    required_skills: List[str] = Field(description="Technical skills required")
    experience_required: int = Field(description="Years of experience required")
    tools: List[str] = Field(description="Tools and technologies mentioned")
    soft_skills: List[str] = Field(description="Soft skills required")


job_parser = PydanticOutputParser(pydantic_object=JobDetails)

job_prompt = PromptTemplate(
    template="""
Extract structured job details from the job description below.

{format_instructions}

Job Description:
{job_description}
""",
    input_variables=["job_description"],
    partial_variables={"format_instructions": job_parser.get_format_instructions()}
)

job_chain = job_prompt | llm | job_parser


# ==============================
# FEATURE 2: Resume Improvement Suggestions
# ==============================

class ResumeSuggestions(BaseModel):
    missing_skills: List[str]
    improvement_points: List[str]
    overall_fit_summary: str


resume_parser = PydanticOutputParser(pydantic_object=ResumeSuggestions)

resume_prompt = PromptTemplate(
    template="""
You are an AI career assistant.

Job Details:
{job_details}

Candidate Resume:
{resume}

Analyze the resume and provide structured suggestions.

{format_instructions}
""",
    input_variables=["job_details", "resume"],
    partial_variables={"format_instructions": resume_parser.get_format_instructions()}
)

resume_chain = resume_prompt | llm | resume_parser


# ==============================
# FEATURE 3: Cover Letter Generator
# ==============================

cover_prompt = PromptTemplate(
    template="""
Write a professional and concise cover letter for the job below.

Job Title: {job_title}
Required Skills: {skills}
Experience Required: {experience} years

Candidate Resume:
{resume}

The cover letter must be formal and well-structured.
""",
    input_variables=["job_title", "skills", "experience", "resume"]
)

cover_chain = cover_prompt | llm | StrOutputParser()


# ==============================
# MAIN FUNCTION
# ==============================

def run_ai_job_assistant(job_description, resume_text):

    # ------------------------------
    # Feature 1: Extract Job Details
    # ------------------------------
    job_details = job_chain.invoke({
        "job_description": job_description
    })

    print("\n===== Extracted Job Details =====")

    print(f"\nJob Title: {job_details.job_title}")

    print("\nRequired Skills:")
    for skill in job_details.required_skills:
        print(f"- {skill}")

    print(f"\nExperience Required: {job_details.experience_required} years")

    print("\nTools:")
    for tool in job_details.tools:
        print(f"- {tool}")

    print("\nSoft Skills:")
    for soft in job_details.soft_skills:
        print(f"- {soft}")


    # ------------------------------
    # Feature 2: Resume Suggestions
    # ------------------------------
    resume_suggestions = resume_chain.invoke({
        "job_details": job_details,
        "resume": resume_text
    })

    print("\n===== Resume Suggestions =====")

    print("\nMissing Skills:")
    if resume_suggestions.missing_skills:
        for skill in resume_suggestions.missing_skills:
            print(f"- {skill}")
    else:
        print("None")

    print("\nImprovement Points:")
    for point in resume_suggestions.improvement_points:
        print(f"- {point}")

    print("\nOverall Fit Summary:")
    print(resume_suggestions.overall_fit_summary)


    # ------------------------------
    # Feature 3: Cover Letter
    # ------------------------------
    cover_letter = cover_chain.invoke({
        "job_title": job_details.job_title,
        "skills": job_details.required_skills,
        "experience": job_details.experience_required,
        "resume": resume_text
    })

    print("\n===== Generated Cover Letter =====\n")
    print(cover_letter)


# ==============================
# USER INPUT SECTION
# ==============================

if __name__ == "__main__":

    print("===================================")
    print("     AI Job Application Assistant")
    print("===================================")

    print("\nPaste Job Description (Press ENTER once to finish):")

    job_lines = []
    while True:
        line = input()
        if line == "":
            break
        job_lines.append(line)

    job_description = "\n".join(job_lines).strip()

    print("\nPaste Resume Text (Press ENTER once to finish):")

    resume_lines = []
    while True:
        line = input()
        if line == "":
            break
        resume_lines.append(line)

    resume_text = "\n".join(resume_lines).strip()

    run_ai_job_assistant(job_description, resume_text)