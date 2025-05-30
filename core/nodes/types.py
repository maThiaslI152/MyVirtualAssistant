# nodes/types.py

from typing import TypedDict

class ChatState(TypedDict):
    input: str
    embedding_model_name: str
    docs: list