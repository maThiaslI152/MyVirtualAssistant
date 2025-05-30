import tiktoken

MAX_TOTAL_TOKENS = 4096
DEFAULT_MAX_RESPONSE_TOKENS = 1024

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_safe_max_tokens(prompt: str, model="gpt-3.5-turbo") -> int:
    prompt_tokens = count_tokens(prompt, model)
    available = MAX_TOTAL_TOKENS - prompt_tokens
    return min(available, DEFAULT_MAX_RESPONSE_TOKENS)
