# benchmark/evaluate_reports.py
import os
import importlib.util
import pandas as pd
from backend.knowledge_base import KnowledgeBase
from utils.pdf_utils import extract_text_from_pdf
from utils.grader import grade_report

# Load queries from queries.py
spec = importlib.util.spec_from_file_location("queries", "benchmark/queries.py")
queries_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queries_module)
queries = queries_module.queries

AGENTS = ["chatgpt", "gemini", "perplexity"]
EVAL_PDF_DIR = "benchmark/evaluation_pdfs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Or load from config

kb = KnowledgeBase()

# STAGE 2: Index PDFs
for topic in queries:
    topic_folder = os.path.join(EVAL_PDF_DIR, topic)
    for agent in AGENTS:
        pdf_path = os.path.join(topic_folder, f"{agent}.pdf")
        if not os.path.exists(pdf_path):
            print(f"Missing: {pdf_path}")
            continue
        text = extract_text_from_pdf(pdf_path)
        metadata = {"topic": topic, "source": agent, "filename": f"{agent}.pdf"}
        kb.add_document(text, metadata)

# STAGE 3 & 4: Query and Grade
results = []
for topic in queries:
    docs = kb.query_similar_documents(topic, top_k=3)
    for doc in docs:
        agent = doc.metadata["source"]
        score, justification = grade_report(topic, doc.page_content)
        results.append({
            "query": topic,
            "agent": agent,
            "score": score,
            "justification": justification
        })

# STAGE 5: Export
df = pd.DataFrame(results)
df.to_csv("benchmark/grading_results.csv", index=False)
print("Exported results to benchmark/grader.csv")