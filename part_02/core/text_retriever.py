# core/text_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import yaml


@dataclass(frozen=True)
class TextHit:
    doc_path: str
    score: int
    snippets: List[str]


@dataclass(frozen=True)
class TextResult:
    query_type: str
    answer_hint: str
    evidence: str
    hits: List[TextHit]
    metrics: Dict[str, Any]


class TextRetriever:
    """
    Unstructured text retriever for Part 2.

    Goals:
    - Find relevant product page(s) in data/unstructured/*.txt
    - Return concise evidence snippets (for LLM grounding)
    - Work with ONLY local files (no web)
    - Keep implementation simple: keyword scoring + snippet extraction

    Supports test-style questions:
      Q3: "key features of <product>"
      Q4: "customers rate <product> in terms of ease of cleaning"
      Q5/Q6: "best reviews / highly rated / recommend" -> rank by review sentiment-ish cues

    NOTE: We do not have explicit numeric ratings in the prompt. So "best reviews"
    is approximated by counting positive/negative cue words in review snippets.
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        data_cfg = self._settings.get("data", {}) or {}
        text_dir = data_cfg.get("text_dir")
        if not text_dir:
            raise ValueError("settings.yaml missing data.text_dir")

        self.text_dir = Path(text_dir)
        if not self.text_dir.exists():
            # allow running from different cwd: resolve relative to project root
            self.text_dir = (self.settings_yaml_path.parent.parent / text_dir).resolve()

        if not self.text_dir.exists():
            raise FileNotFoundError(f"text_dir not found: {self.text_dir}")

        text_cfg = ((self._settings.get("retrieval") or {}).get("text") or {})
        self.top_k_default = int(text_cfg.get("top_k", 3))
        self.max_chunk_chars = int(text_cfg.get("max_chunk_chars", 2000))
        self.keyword_window = int(text_cfg.get("keyword_window", 3))

        # Basic sentiment-ish cue words for "best reviews" tasks
        self.positive_cues = [
            "love", "great", "excellent", "amazing", "perfect", "fantastic", "good",
            "works well", "high quality", "recommend", "easy", "comfortable", "worth"
        ]
        self.negative_cues = [
            "bad", "terrible", "poor", "broken", "disappointed", "hate", "worse",
            "doesn't work", "hard", "difficult", "return", "refund", "problem"
        ]

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    # ---------------------------
    # Public API
    # ---------------------------

    def run(self, question: str, *, top_k: Optional[int] = None) -> TextResult:
        q = question.strip()
        k = int(top_k or self.top_k_default)

        docs = self._list_docs()
        if not docs:
            return TextResult(
                query_type="no_docs",
                answer_hint="No unstructured documents found.",
                evidence=f"No .txt files found under {self.text_dir}",
                hits=[],
                metrics={"text_dir": str(self.text_dir)},
            )

        # Determine query type and keywords
        qtype, keywords = self._infer_query_type_and_keywords(q)

        # Score each document
        scored: List[Tuple[int, Path, List[str]]] = []
        for p in docs:
            text = self._read_text(p)
            score, snippets = self._score_doc_and_snippets(text, keywords, qtype=qtype)
            if score > 0:
                scored.append((score, p, snippets))

        # If nothing matched, fallback to weaker token-based matching
        if not scored:
            fallback_keywords = self._fallback_keywords(q)
            for p in docs:
                text = self._read_text(p)
                score, snippets = self._score_doc_and_snippets(text, fallback_keywords, qtype="fallback")
                if score > 0:
                    scored.append((score, p, snippets))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        hits: List[TextHit] = []
        for score, p, snippets in top:
            hits.append(TextHit(doc_path=str(p), score=score, snippets=snippets[:8]))

        evidence = self._format_evidence(hits)
        answer_hint = self._answer_hint(qtype, hits)

        metrics = {
            "query_type": qtype,
            "requested_top_k": k,
            "matched_docs": len(scored),
            "returned_docs": len(hits),
            "text_dir": str(self.text_dir),
            "keywords": keywords,
        }

        return TextResult(
            query_type=qtype,
            answer_hint=answer_hint,
            evidence=evidence,
            hits=hits,
            metrics=metrics,
        )

    # ---------------------------
    # Doc handling
    # ---------------------------

    def _list_docs(self) -> List[Path]:
        return sorted(self.text_dir.glob("*.txt"))

    def _read_text(self, path: Path) -> str:
        # Read and hard-truncate to avoid huge prompts
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if len(raw) <= self.max_chunk_chars:
            return raw
        return raw[: self.max_chunk_chars] + "\n...[truncated]...\n"

    # ---------------------------
    # Query understanding
    # ---------------------------

    def _infer_query_type_and_keywords(self, q: str) -> Tuple[str, List[str]]:
        q_low = q.lower()

        # Product name heuristic: quoted phrase or capitalized sequence
        # We'll also include all significant tokens as keywords.
        product_phrase = self._extract_product_phrase(q)

        if any(w in q_low for w in ["key features", "features", "specifications", "specs"]):
            qtype = "features"
            keywords = ["features", "feature", "spec", "specifications"]
            if product_phrase:
                keywords += [product_phrase]
        elif any(w in q_low for w in ["review", "reviews", "customers", "feedback", "rate"]):
            qtype = "reviews"
            keywords = ["review", "reviews", "customer", "customers", "feedback"]
            if "clean" in q_low:
                qtype = "reviews_cleaning"
                keywords += ["clean", "cleaning", "easy to clean", "cleanup", "wash"]
            if product_phrase:
                keywords += [product_phrase]
        elif any(w in q_low for w in ["best", "highly rated", "recommend", "top product"]):
            qtype = "best_reviews"
            keywords = ["review", "reviews", "recommend", "rating", "rated", "customer"]
        else:
            qtype = "generic"
            keywords = []
            if product_phrase:
                keywords.append(product_phrase)

        # Add token keywords (avoid very short)
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", q_low) if len(t) >= 4]
        # keep a small set
        keywords += tokens[:8]

        # Dedupe
        keywords = self._dedupe_keep_order([k.strip() for k in keywords if k.strip()])
        return qtype, keywords

    @staticmethod
    def _extract_product_phrase(q: str) -> Optional[str]:
        # Prefer quoted product name
        m = re.search(r"\"([^\"]{3,80})\"", q)
        if m:
            return m.group(1).strip().lower()

        # Heuristic: look for Title Case sequences (e.g., Wireless Bluetooth Headphones)
        # This is imperfect but helpful for the provided test questions.
        words = q.strip().split()
        title_seq: List[str] = []
        for w in words:
            if w[:1].isupper() and w[1:].islower() and len(w) >= 3:
                title_seq.append(w)
            elif w.isupper() and len(w) >= 3:
                title_seq.append(w)
            else:
                if len(title_seq) >= 2:
                    break
                title_seq = []
        if len(title_seq) >= 2:
            return " ".join(title_seq).lower()

        return None

    @staticmethod
    def _fallback_keywords(q: str) -> List[str]:
        q_low = q.lower()
        toks = [t for t in re.split(r"[^a-z0-9]+", q_low) if len(t) >= 4]
        return toks[:10]

    @staticmethod
    def _dedupe_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    # ---------------------------
    # Scoring + snippets
    # ---------------------------

    def _score_doc_and_snippets(self, text: str, keywords: List[str], *, qtype: str) -> Tuple[int, List[str]]:
        """
        Keyword-based scoring:
        - Each keyword hit contributes to score
        - Extract snippet windows around hit lines
        - For best_reviews, add sentiment-ish bonus based on cue words
        """
        if not text.strip() or not keywords:
            # For best_reviews we can still score by sentiment cues alone
            if qtype == "best_reviews":
                return self._sentiment_score(text), self._snippets_for_sentiment(text)
            return 0, []

        lines = text.splitlines()
        hits: List[int] = []
        score = 0

        for i, line in enumerate(lines):
            line_low = line.lower()
            for kw in keywords:
                kw_low = kw.lower()
                if not kw_low:
                    continue
                if kw_low in line_low:
                    score += 2
                    hits.append(i)

        # Extra boost for "features" if we hit likely section headers
        if qtype == "features":
            for i, line in enumerate(lines):
                if any(h in line.lower() for h in ["features", "specifications", "highlights"]):
                    score += 3

        # Sentiment-ish bonus for best_reviews / reviews
        if qtype in ("best_reviews", "reviews", "reviews_cleaning"):
            score += self._sentiment_score(text)

        snippets = self._extract_snippets(lines, hits, window=self.keyword_window)
        return score, snippets

    def _sentiment_score(self, text: str) -> int:
        t = text.lower()
        pos = sum(t.count(w) for w in self.positive_cues)
        neg = sum(t.count(w) for w in self.negative_cues)
        # Score is clipped so it doesn't dominate keyword evidence
        return max(min(pos - neg, 12), -12)

    def _snippets_for_sentiment(self, text: str) -> List[str]:
        # crude: return the first few lines containing positive cues
        lines = text.splitlines()
        hits = []
        for i, line in enumerate(lines):
            ll = line.lower()
            if any(w in ll for w in self.positive_cues):
                hits.append(i)
        return self._extract_snippets(lines, hits, window=self.keyword_window)

    @staticmethod
    def _extract_snippets(lines: List[str], hit_idxs: List[int], window: int) -> List[str]:
        if not hit_idxs:
            return []
        # Merge nearby hits
        hit_idxs = sorted(set(hit_idxs))
        ranges: List[Tuple[int, int]] = []
        for idx in hit_idxs:
            start = max(idx - window, 0)
            end = min(idx + window, len(lines) - 1)
            ranges.append((start, end))

        # Merge overlapping ranges
        merged: List[Tuple[int, int]] = []
        for s, e in sorted(ranges):
            if not merged:
                merged.append((s, e))
            else:
                ps, pe = merged[-1]
                if s <= pe + 1:
                    merged[-1] = (ps, max(pe, e))
                else:
                    merged.append((s, e))

        snippets: List[str] = []
        for s, e in merged[:12]:
            chunk = "\n".join(lines[s : e + 1]).strip()
            if chunk:
                snippets.append(chunk)
        return snippets

    # ---------------------------
    # Formatting output evidence
    # ---------------------------

    @staticmethod
    def _format_evidence(hits: List[TextHit]) -> str:
        if not hits:
            return "No relevant text evidence found."
        parts: List[str] = []
        for i, h in enumerate(hits, 1):
            parts.append(f"--- Text Evidence {i} ---")
            parts.append(f"file: {h.doc_path}")
            parts.append(f"score: {h.score}")
            for j, snip in enumerate(h.snippets, 1):
                parts.append(f"[snippet {j}]")
                parts.append(snip)
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _answer_hint(qtype: str, hits: List[TextHit]) -> str:
        if not hits:
            return "No matching product pages found."
        if qtype == "features":
            return "Summarize the key features/specifications from the most relevant product page(s)."
        if qtype in ("reviews", "reviews_cleaning"):
            return "Summarize what customers say, citing review snippets relevant to the question."
        if qtype == "best_reviews":
            return "Identify the product(s) with the strongest positive review signals and cite snippets."
        return "Use the retrieved snippets to answer the question."

    # ---------------------------
    # Extra helpers (optional)
    # ---------------------------


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    tr = TextRetriever(base / "config" / "settings.yaml")

    qs = [
        "What are the key features of the Wireless Bluetooth Headphones?",
        "How do customers rate the Air Fryer in terms of ease of cleaning?",
        "Which product has the best customer reviews?",
    ]
    for q in qs:
        r = tr.run(q)
        print("\nQ:", q)
        print(r.query_type)
        print(r.answer_hint)
        print(r.evidence[:1200])