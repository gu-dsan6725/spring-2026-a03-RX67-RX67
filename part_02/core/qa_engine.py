# core/qa_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.formatter import FormattedPrompt


@dataclass(frozen=True)
class QAResponse:
    answer: str
    model: str
    usage: Dict[str, Any]
    meta: Dict[str, Any]


class QAEngine:
    """
    LLM caller for Part 2 (multi-source RAG).

    Default backend: LiteLLM (recommended for coursework portability).
    - If litellm is installed, this works.
    - Otherwise, replace _call_llm() with your provider SDK.

    settings.yaml -> llm:
      model, temperature, max_tokens, require_citations, forbid_hallucination
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        llm_cfg = self._settings.get("llm", {}) or {}
        self.model = str(llm_cfg.get("model", "gpt-4o-mini"))
        self.temperature = float(llm_cfg.get("temperature", 0))
        self.max_tokens = int(llm_cfg.get("max_tokens", 1000))
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

    def answer(self, prompt: FormattedPrompt) -> QAResponse:
        """
        Calls the model with system+user messages and returns the answer.
        """
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        raw = self._call_llm(messages)

        text = self._extract_text(raw)
        usage = self._extract_usage(raw)

        # Minimal guardrail: warn if citations are required but absent
        if self.require_citations:
            text = self._ensure_mentions_sources(text)

        return QAResponse(
            answer=text,
            model=self.model,
            usage=usage,
            meta=prompt.meta,
        )

    # ---------------------------
    # LLM backend (LiteLLM)
    # ---------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            from litellm import completion  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LiteLLM is not installed or failed to import. "
                "Install it (uv add litellm) or replace QAEngine._call_llm() "
                "with your provider SDK."
            ) from e

        resp = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if isinstance(resp, dict):
            return resp
        return {"raw": resp}

    # ---------------------------
    # Response parsing
    # ---------------------------

    @staticmethod
    def _extract_text(resp: Dict[str, Any]) -> str:
        try:
            choices = resp.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
        except Exception:
            pass

        # fallback keys
        for k in ("output_text", "content", "text"):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return str(resp).strip()

    @staticmethod
    def _extract_usage(resp: Dict[str, Any]) -> Dict[str, Any]:
        usage = resp.get("usage")
        return usage if isinstance(usage, dict) else {}

    # ---------------------------
    # Guardrails
    # ---------------------------

    def _ensure_mentions_sources(self, text: str) -> str:
        """
        For this assignment, we want the answer to explicitly reference CSV/Text evidence.
        This is intentionally weak (we don't want to be brittle).
        """
        low = text.lower()
        mentions_csv = ("csv" in low) or ("structured evidence" in low)
        mentions_text = ("text evidence" in low) or ("unstructured evidence" in low)

        # If the route is BOTH, we want at least both mentions.
        # If the route is single-source, at least one mention is fine.
        source = str((self._settings.get("meta_source_override") or "")).lower()  # optional hook
        # better: infer from prompt meta if present; we only have final text here, so keep simple.

        if not (mentions_csv or mentions_text):
            return text.rstrip() + "\n\n[warning] No explicit evidence-source mention found (CSV/Text). Consider tightening the prompt."
        return text


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    engine = QAEngine(base / "config" / "settings.yaml")

    demo = FormattedPrompt(
        system="You must cite evidence.",
        user="[Question]\nWhat is total revenue?\n\n[Structured Evidence: CSV]\n...",
        meta={"route": "csv_only"},
    )
    # Requires API key / provider setup
    # print(engine.answer(demo).answer)
    print("QAEngine initialized. (Demo call commented out.)")