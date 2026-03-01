# ============================================
# AI TEXT TRANSFORMER (FULL OLLAMA VERSION)
# ============================================

import requests

print("Connecting to Ollama (LLaMA 3)...\n")

def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Ollama connection failed: {e}"


# --------------------------------------------
# Input Validation
# --------------------------------------------
MAX_WORDS = 300
MIN_WORDS = 100

while True:
    print(f"\nEnter a paragraph between {MIN_WORDS} and {MAX_WORDS} words:\n")
    paragraph = input()

    word_count = len(paragraph.split())

    if MIN_WORDS <= word_count <= MAX_WORDS:
        print(f"\nAccepted ✅ ({word_count} words)\n")
        break
    else:
        print(f"\n⚠ Your paragraph has {word_count} words.")
        print(f"Please enter between {MIN_WORDS} and {MAX_WORDS} words.\n")

print("\n--- AI Text Transformer Output ---\n")


# --------------------------------------------
# 1️⃣ Summary
# --------------------------------------------
summary_prompt = f"""
Provide a concise 2-3 sentence academic summary covering:
- main benefits
- risks
- future outlook

Paragraph:
{paragraph}
"""

print("Generating summary...\n")
summary = ask_ollama(summary_prompt)

print("1. Summary (2–3 sentences):\n")
print(summary)


# --------------------------------------------
# 2️⃣ Tone Detection
# --------------------------------------------
tone_prompt = f"""
Analyze the tone of the following paragraph.
Choose ONE from:
Optimistic, Pessimistic, Neutral, Critical, Analytical, Concerned, Encouraging, Informative, Reflective, Assertive.

Paragraph:
{paragraph}

Answer with only the tone word.
"""

print("\nDetecting tone...\n")
tone = ask_ollama(tone_prompt)

print("2. Detected Tone:\n")
print(tone.strip())


# --------------------------------------------
# 3️⃣ Technical Rewrite
# --------------------------------------------
rewrite_prompt = f"""
You are a professional academic editor.

Rewrite the following paragraph in a highly formal, technical, and scholarly style.

Requirements:
- 170 to 220 words
- Maintain original meaning
- Significantly restructure sentences
- Use advanced academic vocabulary
- Do NOT summarize

Paragraph:
{paragraph}
"""

print("\nGenerating technical rewrite...\n")
rewrite = ask_ollama(rewrite_prompt)

print("3. Improved Technical Version:\n")
print(rewrite)

print("\n✅ Process Completed Successfully!\n")