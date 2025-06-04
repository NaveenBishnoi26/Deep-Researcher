import openai
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import os
load_dotenv()

# Set API keys
openai.api_key = openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key= genai.configure(api_key=os.getenv("GOOGLE_API_KEY")))

# --- ChatGPT (OpenAI GPT-4) ---
def call_chatgpt(query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful research assistant. Write a literature review based on the topic given in queries. It needs to be a moderately detailed generated report. There needs to be an introduction, main content, conclusion, references and other terminologies a usual research paper review might include."},
            {"role": "user", "content": query}
        ],
        temperature=0.7, max_tokens=700
    )
    return response.choices[0].message.content.strip()

# --- Google Gemini ---
def call_gemini(query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(
        f"Write a literature review on: {query}",
        generation_config={"temperature": 0.7, "max_output_tokens": 700}
    )
    return response.text.strip()

# --- LLaMA (OpenRouter / HuggingFace / Local) ---
def call_perplexity(query):
    headers = {
        "Authorization": "2300900a-e198-4501-9125-974bed3af867",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama-3-70b",
        "messages": [
            {"role": "system", "content": "You are a research assistant. Write a literature review."},
            {"role": "user", "content": query}
        ],
        "temperature": 0.7,
        "max_tokens": 700
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    return response.json()['choices'][0]['message']['content'].strip()

# --- Your Own Agent ---
def call_your_agent(query):
    response = requests.post("http://localhost:5000/generate", json={"query": query})
    return response.json()['output']
