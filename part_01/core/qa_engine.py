# core/qa_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List
import os

import yaml

# Ensure .env is loaded before any LLM call (in case main.py load ran too late)
def _ensure_env_loaded() -> None:
    try:
        from dotenv import load_dotenv
        _base = Path(__file__).resolve().parents[1]   # part_01/core -> part_01
        _root = _base.parent                            # repo root
        load_dotenv(_root / ".env")
        load_dotenv(_base / ".env")  # part_01/.env can override (e.g. GROQ_API_KEY)
    except ImportError:
        pass

from core.formatter import FormattedPrompt


@dataclass(frozen=True)
class QAResponse:
    answer: str
    model: str
    usage: Dict[str, Any]
    meta: Dict[str, Any]


class QAEngine:
    """
    LLM wrapper for producing grounded answers from a formatted prompt.

    Recommended backend for this assignment: LiteLLM (supports many providers with one API).
    - If litellm is installed, we'll use it.
    - Otherwise, you can replace _call_llm() with your preferred SDK.

    Settings:
      config/settings.yaml -> llm:
        model, temperature, max_tokens, require_citations, forbid_hallucination

    Environment:
      - For LiteLLM, set provider-specific env vars (e.g., OPENAI_API_KEY).
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        llm_cfg = self._settings.get("llm", {}) or {}
        self.model = str(llm_cfg.get("model", "gpt-4o-mini"))
        # Fallback: if config says OpenAI but only GROQ_API_KEY is set, use Groq
        _ensure_env_loaded()
        if ("gpt" in self.model.lower() or "openai" in self.model.lower()) and not os.environ.get("OPENAI_API_KEY") and os.environ.get("GROQ_API_KEY"):
            self.model = "groq/llama-3.3-70b-versatile"
        self.temperature = float(llm_cfg.get("temperature", 0))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1200))
        self.require_citations = bool(llm_cfg.get("require_citations", True))
        self.forbid_hallucination = bool(llm_cfg.get("forbid_hallucination", True))

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def answer(self, formatted: FormattedPrompt) -> QAResponse:
        """
        Send the formatted prompt to the model and return an answer.
        """
        messages = self._build_messages(formatted.prompt)

        raw = self._call_llm(messages)

        text = self._extract_text(raw)
        usage = self._extract_usage(raw)

        # Optional post-check: if citations required, enforce minimally
        if self.require_citations:
            text = self._ensure_citations_or_warn(text)

        return QAResponse(
            answer=text,
            model=self.model,
            usage=usage,
            meta=formatted.meta,
        )

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        """
        Chat-style messages. Keeps it compatible with most providers.
        """
        # You can separate system/user, but many providers treat a single user prompt fine.
        # We'll keep a lightweight system message to reinforce grounding.
        sys_lines = [
            "You are a careful codebase QA assistant.",
            "Use only the provided evidence. Do not guess file paths.",
        ]
        if self.forbid_hallucination:
            sys_lines.append("If evidence is missing, say what is missing and what to search next.")
        if self.require_citations:
            sys_lines.append("Cite file paths for key claims (prefer path:line).")

        return [
            {"role": "system", "content": "\n".join(sys_lines)},
            {"role": "user", "content": prompt},
        ]

    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Calls LiteLLM if available.

        Returns a provider-agnostic dict-like response.
        """
        _ensure_env_loaded()

        try:
            from litellm import completion  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LiteLLM is not available. Install it (pip/uv add litellm) "
                "or replace QAEngine._call_llm() with your provider SDK."
            ) from e

        # Pass api_key explicitly so .env is respected even when run from different cwd
        api_key = None
        key_name = ""
        if self.model.startswith("groq/"):
            api_key = os.environ.get("GROQ_API_KEY")
            key_name = "GROQ_API_KEY"
        elif "gpt" in self.model.lower() or "openai" in self.model.lower():
            api_key = os.environ.get("OPENAI_API_KEY")
            key_name = "OPENAI_API_KEY"

        if not api_key and key_name:
            raise RuntimeError(
                f"Model {self.model!r} requires {key_name}. "
                f"Set it in .env at repo root or export it before running."
            )

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if api_key:
            kwargs["api_key"] = api_key

        resp = completion(**kwargs)

        # LiteLLM returns a pydantic-ish object; convert to plain dict if needed
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if isinstance(resp, dict):
            return resp
        return {"raw": resp}

    @staticmethod
    def _extract_text(resp: Dict[str, Any]) -> str:
        """
        Extract assistant message content across common response shapes.
        """
        # OpenAI/LiteLLM common shape:
        # resp["choices"][0]["message"]["content"]
        try:
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
        except Exception:
            pass

        # Fallback: try other shapes
        for k in ("output_text", "content", "text"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # Last resort
        return str(resp).strip()

    @staticmethod
    def _extract_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
        usage = resp.get("usage")
        if isinstance(usage, dict):
            return usage
        return {}

    def _ensure_citations_or_warn(self, text: str) -> str:
        """
        Minimal citation enforcement: check whether the answer contains at least one
        file-ish reference (e.g., something.py or pyproject.toml or Dockerfile).

        This is NOT perfect; it's a guardrail so you notice when the model ignored citations.
        """
        # Very lightweight heuristic
        has_file = any(
            token in text
            for token in ("pyproject.toml", "package.json", "Dockerfile", ".py:", ".py", ".ts", ".yaml", ".yml")
        )
        if has_file:
            return text

        warning = (
            "\n\n[warning] The answer contains no obvious file citations. "
            "Consider tightening the prompt or increasing evidence quality."
        )
        return text.strip() + warning


# Optional quick manual test (won't work without API keys):
if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    engine = QAEngine(base / "config" / "settings.yaml")

    fp = FormattedPrompt(
        prompt="[Question]\nWhat dependencies?\n\n[Evidence]\npyproject.toml:12:dependencies = ...",
        meta={"route": "deps"},
    )
    print(engine.answer(fp).answer)