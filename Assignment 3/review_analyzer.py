import json
from typing import List
from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# ---------------------------
# 1️⃣ Pydantic Model
# ---------------------------
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: Positive, Negative, or Neutral")
    rating: int = Field(description="Rating from 1 to 5")
    key_features: List[str] = Field(description="Important product features mentioned")
    improvement_suggestions: List[str] = Field(description="Suggestions for improvement")


# ---------------------------
# 2️⃣ Parser Setup
# ---------------------------
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)
format_instructions = parser.get_format_instructions()


prompt = PromptTemplate(
    template="""
You are a strict product review analyzer.

Analyze the review carefully and extract:

1. Overall sentiment (Positive, Negative, or Neutral)
2. Rating (integer from 1 to 5)
3. All important product features mentioned (both positive and negative)
4. Clear improvement suggestions based on complaints

IMPORTANT RULES:
- Extract ALL major features mentioned.
- If a feature is criticized, still include it in key_features.
- improvement_suggestions must be short phrases, not sentences.
- rating must be an integer between 1 and 5.

{format_instructions}

Review:
{review_text}
""",
    input_variables=["review_text"],
    partial_variables={"format_instructions": format_instructions},
)

# ---------------------------
# 4️⃣ Ollama LLM (Offline)
# ---------------------------
llm = ChatOllama(
    model="phi3:mini",
    temperature=0
)


# ---------------------------
# 5️⃣ Analysis Function
# ---------------------------
def analyze_review(review_text: str):
    try:
        formatted_prompt = prompt.format(review_text=review_text)
        response = llm.invoke(formatted_prompt)

        structured_output = parser.parse(response.content)

        return structured_output

    except ValidationError as ve:
        print("\n❌ Validation Error:")
        print(ve)
        return None

    except Exception as e:
        print("\n❌ Unexpected Error:")
        print(e)
        return None


# ---------------------------
# 6️⃣ Interactive Input
# ---------------------------
if __name__ == "__main__":

    print("Paste product review below.")
    print("Press ENTER twice when finished.\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    review_text = "\n".join(lines)

    result = analyze_review(review_text)

    if result:
        print("\n✅ Review Analysis:\n")
        print(f"Sentiment: {result.sentiment}")
        print(f"Rating: {result.rating}/5\n")

        print("Key Features:")
        for feature in result.key_features:
            print(f"- {feature}")

        print("\nImprovement Suggestions:")
        for suggestion in result.improvement_suggestions:
            print(f"- {suggestion}")
    else:
        print("\n⚠️ Failed to extract structured data.")