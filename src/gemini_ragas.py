"""Preserve RAGAS sample counts when a Gemini model only supports n=1."""
from ragas.llms.base import LangchainLLMWrapper
from langchain_core.outputs import Generation, LLMResult
from pathlib import Path
import hashlib
import json
import asyncio
import uuid
from src.gemini_rate_limit import wait_for_gemini


class SingleCandidateLLMWrapper(LangchainLLMWrapper):
    def set_run_config(self, run_config):
        super().set_run_config(run_config)
        from openai import RateLimitError, InternalServerError, APITimeoutError, APIConnectionError
        self.run_config.exception_types = (RateLimitError, InternalServerError,
                                           APITimeoutError, APIConnectionError)

    def _cache_path(self, prompt, temperature, stop, index):
        key = ["ragas-sample-v1", getattr(self.langchain_llm, "model_name", "unknown"),
               prompt.to_string() if hasattr(prompt, "to_string") else str(prompt),
               temperature, stop, index]
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()
        return Path(__file__).resolve().parent.parent / ".cache" / "ragas" / (digest + ".json")

    def _read(self, path):
        try:
            if path.exists():
                return Generation(text=json.loads(path.read_text(encoding="utf-8"))["text"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _save(self, path, sample):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps({"text": sample.text}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if n < 1:
            raise ValueError("n must be positive")
        temperature = self.get_temperature(n) if temperature is None else temperature
        samples = []
        for index in range(n):
            path = self._cache_path(prompt, temperature, stop, index)
            sample = self._read(path)
            if sample is None:
                wait_for_gemini(self.langchain_llm.model_name)
                result = self.langchain_llm.generate_prompt(
                    prompts=[prompt], n=1, temperature=temperature, stop=stop, callbacks=callbacks)
                sample = result.generations[0][0]
                self._save(path, sample)
            samples.append(sample)
        return LLMResult(generations=[samples])

    async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
        if n < 1:
            raise ValueError("n must be positive")
        temperature = self.get_temperature(n) if temperature is None else temperature
        samples = []
        for index in range(n):
            path = self._cache_path(prompt, temperature, stop, index)
            sample = self._read(path)
            if sample is None:
                await asyncio.to_thread(wait_for_gemini, self.langchain_llm.model_name)
                result = await self.langchain_llm.agenerate_prompt(
                    prompts=[prompt], n=1, temperature=temperature, stop=stop, callbacks=callbacks)
                sample = result.generations[0][0]
                self._save(path, sample)
            samples.append(sample)
        return LLMResult(generations=[samples])
