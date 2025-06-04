from benchmark.api_clients import call_chatgpt, call_gemini, call_perplexity, call_your_agent
from benchmark.queries import queries
from benchmark.utils.grader import grade_report
import pandas as pd
from tqdm import tqdm

agents = {
    #"your_agent": call_your_agent,
    "chatgpt": call_chatgpt,
    "gemini": call_gemini,
    "perplexity": call_perplexity
}

grading_key = "OPENAI_API_KEY"  # Can be different from ChatGPT query key

results = {agent: [] for agent in agents}
scores = {agent: [] for agent in agents}

for query in tqdm(queries, desc="Running benchmark"):
    for agent_name, func in agents.items():
        try:
            output = func(query)
            results[agent_name].append(output)
            score = grade_report(query, output)
            scores[agent_name].append(score)
        except Exception as e:
            print(f"{agent_name} failed on: {query[:30]}...: {e}")
            results[agent_name].append("ERROR")
            scores[agent_name].append(-1)

df = pd.DataFrame(scores, index=queries)
df["Average"] = df.mean(axis=1)
df.loc["Average"] = df.mean()
df.to_csv("benchmark/results.csv")
print("\nBenchmarking complete. Results saved to benchmark/results.csv")
