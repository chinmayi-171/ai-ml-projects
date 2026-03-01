from pydantic import BaseModel
from typing import List
from langchain_core.output_parsers import JsonOutputParser


class ResumeSchema(BaseModel):
    name: str
    email: str
    skills: List[str]
    experience_years: int
    education: List[str]


def get_parser():
    return JsonOutputParser(pydantic_object=ResumeSchema)