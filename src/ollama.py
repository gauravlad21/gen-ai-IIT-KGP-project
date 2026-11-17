"""Local Ollama adapter for the dictionary-returning LLMAnswer interface."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaModel:
    def __init__(
        self, model: str, timeout: float = 120, temperature: float = 0, seed: int = 42, think: bool = False
    ):
        if not model.strip():
            raise ValueError("An explicit Ollama model name is required.")
        if timeout <= 0:
            raise ValueError("Ollama timeout must be positive.")
        if temperature < 0:
            raise ValueError("Ollama temperature must not be negative.")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.think = think
        self.last_prompt: str | None = None
        self.last_response: dict | None = None

    def __call__(self, prompt: str) -> dict:
        self.last_prompt = prompt
        self.last_response = None
        request = Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": self.think,  # built-in reasoning would blur the prompting-strategy comparison
                    "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": 8192},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise ValueError(f"Ollama did not finish within {self.timeout:g}s; no answer was accepted.") from exc
        except URLError as exc:
            raise RuntimeError(
                "Cannot reach local Ollama. Start `ollama serve`, pull the selected model, "
                f"and retry. Connection error: {exc}"
            ) from exc
        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama returned an invalid JSON response envelope.") from exc
        if not isinstance(envelope, dict):
            raise ValueError("Ollama response envelope must be an object.")
        self.last_response = envelope
        if envelope.get("error"):
            raise RuntimeError(f"Ollama error: {envelope['error']}")
        if envelope.get("done") is not True:
            raise ValueError("Ollama did not complete generation.")
        if envelope.get("done_reason") == "length":
            raise ValueError("Ollama generation was truncated; no answer was accepted.")
        content = envelope.get("response")
        if not isinstance(content, str):
            raise ValueError("Ollama response is missing generated text.")
        try:
            answer = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Model generated invalid JSON; inspect the saved raw response.") from exc
        if (
            not isinstance(answer, dict)
            or not isinstance(answer.get("answer"), str)
            or not isinstance(answer.get("should_refuse"), bool)
            or not isinstance(answer.get("reason"), str)
            or not isinstance(answer.get("citations"), list)
        ):
            raise ValueError("Model output does not match the answer schema.")
        if any(
            not isinstance(item, dict) or not isinstance(item.get("doc_id"), str)
            or not item["doc_id"].strip()
            for item in answer["citations"]
        ):
            raise ValueError("Model citations must contain nonempty document IDs.")
        if answer["should_refuse"]:
            if answer["citations"] or not answer["reason"].strip():
                raise ValueError("A refusal must have no citations and a nonempty reason.")
        elif not answer["answer"].strip() or not answer["citations"]:
            raise ValueError("A supported answer must contain answer text and citations.")
        return answer
