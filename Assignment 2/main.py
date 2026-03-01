from langchain_ollama import ChatOllama
from prompt_template import get_prompt
from parser_module import get_parser
import json


def extract_resume_data(resume_text):

    # Initialize Ollama Model
    llm = ChatOllama(
        model="phi3:mini",  # You can also use "llama3"
        temperature=0
    )

    prompt = get_prompt()
    parser = get_parser()

    # Create full chain
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"resume_text": resume_text})
        return result
    except Exception as e:
        print("❌ Error parsing output:", e)
        return None


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

    if result:
        print("\n✅ Extracted Resume Data (JSON):\n")
        print(json.dumps(result, indent=4))