# core/combiner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from core.csv_retriever import CSVResult
from core.text_retriever import TextResult


@dataclass(frozen=True)
class CombinedEvidence:
    """
    A single merged evidence bundle to pass into formatter/LLM.
    """
    route: str                 # csv_only | text_only | both
    source: str                # csv | text | both
    question: str

    csv_result: Optional[CSVResult]
    text_result: Optional[TextResult]

    evidence_text: str         # merged + structured for prompt
    meta: Dict[str, Any]


class Combiner:
    """
    Combines CSVResult + TextResult into a single evidence block.

    Principles:
    - Keep numeric computation explicitly separated from text evidence.
    - Make the model's job easy: clearly label where numbers come from.
    - Truncate aggressively to avoid prompt bloat (TextRetriever already truncates docs).
    """

    def __init__(self, *, max_chars: int = 6000):
        self.max_chars = max_chars

    def combine(
        self,
        question: str,
        *,
        route: str,
        source: str,
        csv_result: Optional[CSVResult] = None,
        text_result: Optional[TextResult] = None,
    ) -> CombinedEvidence:
        parts: List[str] = []
        meta: Dict[str, Any] = {"route": route, "source": source}

        parts.append("[Question]")
        parts.append(question.strip())
        parts.append("")

        # Structured evidence
        if csv_result is not None:
            parts.append("[Structured Evidence: CSV]")
            parts.append(f"query_type: {csv_result.query_type}")
            parts.append(f"answer: {csv_result.answer_text}")
            parts.append("evidence:")
            parts.append(csv_result.evidence)
            parts.append("metrics:")
            parts.append(self._pretty_kv(csv_result.metrics))
            parts.append("")
            meta["csv_metrics"] = csv_result.metrics

        # Unstructured evidence
        if text_result is not None:
            parts.append("[Unstructured Evidence: Text]")
            parts.append(f"query_type: {text_result.query_type}")
            parts.append(f"answer_hint: {text_result.answer_hint}")
            parts.append("evidence:")
            parts.append(text_result.evidence)
            parts.append("")
            meta["text_metrics"] = text_result.metrics

        # If neither exists (should not happen), still provide a placeholder
        if csv_result is None and text_result is None:
            parts.append("[Evidence]")
            parts.append("No evidence was produced by retrievers.")
            parts.append("")

        merged = "\n".join(parts).strip()
        merged = self._truncate(merged, self.max_chars)

        meta["evidence_chars"] = len(merged)

        return CombinedEvidence(
            route=route,
            source=source,
            question=question.strip(),
            csv_result=csv_result,
            text_result=text_result,
            evidence_text=merged,
            meta=meta,
        )

    @staticmethod
    def _pretty_kv(d: Dict[str, Any]) -> str:
        lines: List[str] = []
        for k in sorted(d.keys()):
            v = d[k]
            # Keep multi-line values compact
            if isinstance(v, str) and "\n" in v:
                v_show = v.splitlines()[0] + " ... (multiline)"
            else:
                v_show = v
            lines.append(f"- {k}: {v_show}")
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = text[: max_chars - 250]
        tail = text[-200:]
        return f"{head}\n...[truncated {len(text) - max_chars} chars]...\n{tail}"


if __name__ == "__main__":
    # Minimal smoke test (doesn't require actual data)
    dummy_csv = CSVResult(
        query_type="top_region_by_units_sold",
        answer_text="Top region is West with 12345 units.",
        evidence="Grouped by region and summed units_sold.",
        metrics={"top_region": "West", "top_units_sold": 12345},
    )
    dummy_text = TextResult(
        query_type="reviews_cleaning",
        answer_hint="Summarize cleaning-related reviews.",
        evidence="--- Text Evidence 1 ---\nfile: AirFryer.txt\n[snippet 1]\nEasy to clean...\n",
        hits=[],
        metrics={"matched_docs": 1},
    )

    c = Combiner(max_chars=2000)
    out = c.combine(
        "Recommend a product.",
        route="both",
        source="both",
        csv_result=dummy_csv,
        text_result=dummy_text,
    )
    print(out.evidence_text)