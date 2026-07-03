import asyncio
import json
import logging
from typing import Any, Dict

from llama_cpp import Llama


class MacroRegimeClassifier:
    def __init__(self, model_path: str, n_ctx: int = 512):
        # n_gpu_layers=-1 offloads all computation to the Metal GPU natively
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,  # Suppress raw C++ logging to keep our audit trail clean
        )

        # We enforce a strict system prompt to guarantee deterministic JSON output
        self.system_prompt = (
            "You are a high-frequency trading risk classification engine. "
            "Analyze the macroeconomic headline and classify the market regime into one of: "
            "[0: NORMAL, 1: HIGH_VOLATILITY, 2: BEARISH_SHOCK, 3: BULLISH_SHOCK]. "
            'Output ONLY valid JSON in the format: {"regime_id": <int>, "confidence": <float>}.'
        )

    def _predict_sync(self, headline: str) -> Dict[str, Any]:
        """Synchronous inference call to the Metal backend."""
        prompt = f"{self.system_prompt}\nHeadline: {headline}\nOutput JSON:"

        try:
            response = self.llm(
                prompt,
                max_tokens=32,
                stop=[
                    "}"
                ],  # Terminate generation immediately upon JSON closure to compress latency
                temperature=0.0,  # Zero entropy for maximum determinism
            )

            # Reconstruct the JSON (since we stopped at the closing brace)
            text_out = response["choices"][0]["text"] + "}"
            return json.loads(text_out)

        except Exception as e:
            logging.error(f"LLM Classification Failed or Hallucinated: {e}")
            # Graceful degradation: Default to safe NORMAL state if inference fails
            return {"regime_id": 0, "confidence": 0.0}

    async def classify_headline(self, headline: str) -> Dict[str, Any]:
        """
        Asynchronous wrapper. Offloads the blocking LLM inference to a separate
        thread pool so it never stalls the Python orchestrator's event loop.
        """
        return await asyncio.to_thread(self._predict_sync, headline)
