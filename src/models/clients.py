"""
Unified model client for LLMBench.
Wraps OpenAI, Groq, and local HuggingFace models
behind a single interface.
"""

import os
import time
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
import groq

load_dotenv()


@dataclass
class ModelResponse:
    model_id: str
    prompt: str
    response: str
    latency_ms: float
    tokens_used: int
    cost_usd: float


# Cost per 1M tokens (input + output averaged)
MODEL_COSTS = {
    "gpt-4o-mini": 0.15,
    "llama-3.1-8b-instant": 0.0,       # Groq free tier
    "llama-3.3-70b-versatile": 0.0,    # Groq free tier
    "phi-3.5-mini": 0.0,               # local
}

MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "display_name": "GPT-4o-mini",
        "family": "OpenAI",
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "display_name": "Llama 3.1 8B",
        "family": "Meta (via Groq)",
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "display_name": "Llama 3.3 70B",
        "family": "Meta (via Groq)",
    },
}


class ModelClient:
    def __init__(self):
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.groq_client = groq.Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    def call(
        self,
        model_id: str,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Call any model through unified interface."""
        model_info = MODELS.get(model_id, {})
        provider = model_info.get("provider", "openai")

        start = time.time()

        if provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens

        elif provider == "groq":
            response = self.groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens

        else:
            raise ValueError(f"Unknown provider: {provider}")

        latency_ms = (time.time() - start) * 1000
        cost = (tokens / 1_000_000) * MODEL_COSTS.get(model_id, 0)

        return ModelResponse(
            model_id=model_id,
            prompt=prompt,
            response=text,
            latency_ms=round(latency_ms, 1),
            tokens_used=tokens,
            cost_usd=round(cost, 6),
        )

    def call_all_models(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        max_tokens: int = 512,
    ) -> dict[str, ModelResponse]:
        """Call all models and return responses."""
        results = {}
        for model_id in MODELS:
            try:
                results[model_id] = self.call(
                    model_id, prompt, system, max_tokens
                )
                time.sleep(0.5)  # rate limit buffer
            except Exception as e:
                print(f"  Warning: {model_id} failed — {e}")
        return results