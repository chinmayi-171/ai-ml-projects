import json
import re
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from typing import List


# ----- Schema -----
class ResumeSchema(BaseModel):
    name: str
    email: str
    skills: List[str]
    experience_years: int
    education: List[str]


# ----- Extract JSON safely from model output -----
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


# ----- Main Extraction Function -----
def extract_resume_data(resume_text):

    llm = ChatOllama(
        model="phi3:mini",
        temperature=0
    )

    prompt = f"""
Extract information from the following resume text.

Return ONLY valid JSON in this exact format:

{{
  "name": "",
  "email": "",
  "skills": [],
  "experience_years": 0,
  "education": []
}}

IMPORTANT:
- Education must be a list of strings.
- Do not return objects inside education.
- Do not explain anything.
- Return JSON only.

Resume:
{resume_text}
"""

    response = llm.invoke(prompt)
    raw_text = response.content

    json_text = extract_json(raw_text)

    if not json_text:
        print("❌ No valid JSON found in model output.")
        return None

    data = json.loads(json_text)

    # Validate using Pydantic
    validated = ResumeSchema(**data)

    return validated.model_dump()


# ----- Run Program -----
if __name__ == "__main__":

    print("Paste the resume text below.")
    print("When finished, press ENTER twice.\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    resume_text = "\n".join(lines)

    result = extract_resume_data(resume_text)

    print("\n✅ Extracted Resume Data (JSON):\n")
    print(json.dumps(result, indent=4))