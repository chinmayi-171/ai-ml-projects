from langchain_core.prompts import PromptTemplate


def get_prompt():

    template = """
Extract information from the resume below.

Return ONLY valid JSON in this exact format:

{{
  "name": "",
  "email": "",
  "skills": [],
  "experience_years": 0,
  "education": []
}}

Rules:
- Fill actual values
- Do NOT return schema
- Do NOT return properties
- Do NOT explain anything
- Return JSON only

Resume:
{resume_text}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["resume_text"],
    )

    return prompt