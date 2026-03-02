# core/formatter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml

from core.combiner import CombinedEvidence


@dataclass(frozen=True)
class FormattedPrompt:
    system: str
    user: str
    meta: Dict[str, Any]


class Formatter:
    """
    Formats the merged evidence into a prompt for the LLM.

    For Part 2, the key is to force:
    - numeric claims come from CSV evidence
    - descriptive/review claims come from Text evidence
    - if something is missing, say what evidence is missing
    """

    def __init__(self, settings_yaml_path: str | Path, *, prompts_dir: Optional[str | Path] = None):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        llm_cfg = self._settings.get("llm", {}) or {}
        self.require_citations = bool(llm_cfg.get("require_citations", True))
        self.forbid_hallucination = bool(llm_cfg.get("forbid_hallucination", True))

        fmt_cfg = self._settings.get("formatting", {}) or {}
        self.include_sources = bool(fmt_cfg.get("include_sources", True))
        self.include_numeric_evidence = bool(fmt_cfg.get("include_numeric_evidence", True))
        self.max_context_blocks = int(fmt_cfg.get("max_context_blocks", 10))

        self.prompts_dir = Path(prompts_dir) if prompts_dir else (self.settings_yaml_path.parent.parent / "prompts")
        self._system_prompt = self._load_prompt_file("system.txt") or self._default_system_prompt()

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def _load_prompt_file(self, name: str) -> Optional[str]:
        p = self.prompts_dir / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
        return None

    def _default_system_prompt(self) -> str:
        lines = [
            "You are a multi-source RAG assistant.",
            "You must answer using ONLY the provided evidence.",
            "Do not invent products, numbers, or file contents.",
            "",
            "Rules:",
            "- If you cite sales/revenue/units/region results, they MUST come from [Structured Evidence: CSV].",
            "- If you cite features/specs/reviews/customer sentiment, they MUST come from [Unstructured Evidence: Text].",
            "- If evidence is insufficient, say exactly what is missing and what retrieval should be done next.",
        ]
        if self.require_citations:
            lines.append("- Always cite which evidence section you used (CSV vs Text) and mention file names if present.")
        return "\n".join(lines).strip()

    def format(self, combined: CombinedEvidence) -> FormattedPrompt:
        """
        Produces system + user messages.

        The user message contains:
        - the merged evidence
        - explicit response format requirements (compact)
        """
        user_parts: List[str] = []

        # Provide merged evidence directly
        user_parts.append(combined.evidence_text)
        user_parts.append("")

        # Output constraints
        user_parts.append("[Output Requirements]")
        user_parts.append("- Answer the question directly.")
        user_parts.append("- Separate your answer into: (1) Conclusion (2) Evidence (3) Caveats.")
        if self.require_citations:
            user_parts.append("- In Evidence, explicitly reference whether each claim came from CSV or Text evidence.")
        if self.forbid_hallucination:
            user_parts.append("- If you cannot answer part of the question, say what is missing and what file/data to retrieve.")
        user_parts.append("")

        # Optional: If the route is BOTH, enforce integration
        if combined.source == "both":
            user_parts.append("[Integration Requirement]")
            user_parts.append("- Use BOTH CSV and Text evidence. Do not answer using only one source.")
            user_parts.append("")

        user_msg = "\n".join(user_parts).strip()

        meta = dict(combined.meta)
        meta.update({
            "question": combined.question,
        })

        return FormattedPrompt(system=self._system_prompt, user=user_msg, meta=meta)


if __name__ == "__main__":
    # minimal smoke test
    dummy = CombinedEvidence(
        route="both",
        source="both",
        question="Recommend a product.",
        csv_result=None,
        text_result=None,
        evidence_text="[Question]\nRecommend a product.\n\n[Evidence]\nNo evidence.",
        meta={"route": "both", "source": "both"},
    )
    base = Path(__file__).resolve().parents[1]
    f = Formatter(base / "config" / "settings.yaml")
    fp = f.format(dummy)
    print("SYSTEM:\n", fp.system[:400])
    print("\nUSER:\n", fp.user[:600])