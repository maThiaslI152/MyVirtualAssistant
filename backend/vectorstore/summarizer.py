# backend/vectorstore/summarizer.py

from typing import List, Dict, Tuple
import requests
import os
import json

LLM_ENDPOINT = os.getenv("LLM_SUMMARY_URL", "http://localhost:1234/v1/chat/completions")
HEADERS = {"Content-Type": "application/json"}

SYSTEM_PROMPT = "You are a helpful summarizer."


def summarize_messages(messages: List[Dict]) -> Tuple[str, List[str]]:
    chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    prompt = (
        "Summarize the following conversation and generate a list of relevant tags (keywords).\n\n"
        "Return JSON like this:\n"
        "{\"summary\": \"...\", \"tags\": [\"tag1\", \"tag2\"]}\n\n"
        f"{chat_text}"
    )

    payload = {
        "model": "your-model-name",  # Replace as needed
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }

    try:
        response = requests.post(LLM_ENDPOINT, json=payload, headers=HEADERS, timeout=30)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(result)
        return parsed.get("summary", "(summary unavailable)"), parsed.get("tags", [])
    except Exception as e:
        print(f"[!] Summary+tag failed: {e}")
        return "(summary unavailable)", []