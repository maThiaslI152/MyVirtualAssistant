# llm.py

from langchain_openai import ChatOpenAI

# Local LLM via LM Studio (DeepSeek)
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.7,
    streaming=True,
    verbose=True,
)
