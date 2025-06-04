# test_compare_and_grade_topic.py

import os
from utils.pdf_utils import extract_text_from_pdf
from utils.grader import grade_report

# Set your topic and agent list
topic = "Solid"  # Change to any topic you want to test
agents = ["chatgpt", "gemini", "perplexity", "custom_agent"]
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
pdf_dir = os.path.join(current_dir, f"evaluation_pdfs\\{topic}") 
openai_api_key = os.getenv("OPENAI_API_KEY")  # Or paste your key directly

results = []

for agent in agents:
    pdf_path = os.path.join(pdf_dir, f"{agent}.pdf")
    if not os.path.exists(pdf_path):
        print(f"Missing PDF for {agent}: {pdf_path}")
        continue
    print(f"\n--- Grading {agent}.pdf ---")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"No text extracted from {pdf_path}")
        continue
    score, justification = grade_report(topic, text)
    print(f"Agent: {agent}")
    print(f"{score}")
    print(f"Justification: {justification}\n")
    results.append({"agent": agent, "score": score, "justification": justification})

print("\n=== Summary ===")
for r in results:
    print(f"{r['agent']}: Score={r['score']} | Justification={r['justification'][:100]}...")