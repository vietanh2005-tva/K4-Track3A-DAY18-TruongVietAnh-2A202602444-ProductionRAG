"""Retry transient generation errors while respecting the shared Gemini budget."""
import time
from openai import RateLimitError, InternalServerError, APITimeoutError, APIConnectionError
from src.gemini_rate_limit import wait_for_gemini


def create_completion(client, model, **kwargs):
    gemini = "generativelanguage.googleapis.com" in str(client.base_url)
    for attempt in range(4):
        if gemini:
            wait_for_gemini(model)
        try:
            return client.chat.completions.create(model=model, **kwargs)
        except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as exc:
            if attempt == 3 or "requestsperday" in str(exc).lower():
                raise
            time.sleep(min(2 ** attempt, 8))
