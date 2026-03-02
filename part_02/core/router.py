# core/router.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import yaml


@dataclass(frozen=True)
class RouteResult:
    route: str
    source: str            # "csv" | "text" | "both"
    score: int
    matched_keywords: List[str]
    matched_regex: List[str]
    debug: Dict[str, Any]


class Router:
    """
    Multi-source router driven by config/routes.yaml.

    Matching logic (simple + stable):
    - Regex hits are strong signals (+6 each)
    - Keyword hits are weaker (+2 each)
    - Break ties by priority (manual), then score
    - Default fallback: text_only (safe, but you can change to csv_only or both)

    Recommended priority for Part 2:
      both > csv_only > text_only
    because many "recommend/best" questions may also include words like "sales".
    """

    REGEX_HIT = 6
    KEYWORD_HIT = 2

    def __init__(self, routes_yaml_path: str | Path):
        self.routes_yaml_path = Path(routes_yaml_path)
        self._doc = self._load_yaml(self.routes_yaml_path)

        routes = self._doc.get("routes", {})
        if not isinstance(routes, dict) or not routes:
            raise ValueError(f"No routes found in {self.routes_yaml_path}")

        self.routes: Dict[str, Dict[str, Any]] = routes

        # Precompile regex patterns per route
        self._compiled_regex: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for route_name, cfg in self.routes.items():
            match_cfg = cfg.get("match", {}) or {}
            patterns = match_cfg.get("regex", []) or []
            compiled: List[Tuple[str, re.Pattern]] = []
            for p in patterns:
                try:
                    compiled.append((p, re.compile(p, re.IGNORECASE)))
                except re.error as e:
                    raise ValueError(f"Invalid regex in route '{route_name}': {p!r} ({e})") from e
            self._compiled_regex[route_name] = compiled

        # Priority (higher wins). You can also move this into YAML if you prefer.
        self._priority = {
            "both": 30,
            "csv_only": 20,
            "text_only": 10,
        }

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"routes.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"routes.yaml must parse to a dict, got: {type(doc)}")
        return doc

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _match_keywords(q_norm: str, keywords: List[str]) -> List[str]:
        hits: List[str] = []
        for kw in keywords or []:
            kw_norm = kw.lower().strip()
            if kw_norm and kw_norm in q_norm:
                hits.append(kw)
        return hits

    def route(self, question: str) -> RouteResult:
        q_norm = self._normalize(question)

        best: Optional[RouteResult] = None

        for route_name, cfg in self.routes.items():
            source = str(cfg.get("source", "")).strip().lower()
            match_cfg = cfg.get("match", {}) or {}

            keywords = match_cfg.get("keywords", []) or []
            matched_keywords = self._match_keywords(q_norm, keywords)

            matched_regex: List[str] = []
            for raw, pat in self._compiled_regex.get(route_name, []):
                if pat.search(question) or pat.search(q_norm):
                    matched_regex.append(raw)

            score = len(matched_regex) * self.REGEX_HIT + len(matched_keywords) * self.KEYWORD_HIT

            # Skip routes with no signal at all
            if score <= 0:
                continue

            candidate = RouteResult(
                route=route_name,
                source=source if source in ("csv", "text", "both") else "text",
                score=score,
                matched_keywords=matched_keywords,
                matched_regex=matched_regex,
                debug={"priority": self._priority.get(route_name, 0)},
            )

            best = self._better(best, candidate)

        # Fallback: if nothing matched, choose text_only (safe for general questions)
        if best is None:
            fallback = "text_only" if "text_only" in self.routes else next(iter(self.routes.keys()))
            fb_cfg = self.routes.get(fallback, {})
            fb_source = str(fb_cfg.get("source", "text")).lower()
            return RouteResult(
                route=fallback,
                source=fb_source if fb_source in ("csv", "text", "both") else "text",
                score=0,
                matched_keywords=[],
                matched_regex=[],
                debug={"priority": self._priority.get(fallback, 0), "fallback": True},
            )

        return best

    def _better(self, current: Optional[RouteResult], candidate: RouteResult) -> RouteResult:
        if current is None:
            return candidate

        cur_pri = int(current.debug.get("priority", 0))
        cand_pri = int(candidate.debug.get("priority", 0))

        # 1) higher priority wins
        if cand_pri != cur_pri:
            return candidate if cand_pri > cur_pri else current

        # 2) higher score wins
        if candidate.score != current.score:
            return candidate if candidate.score > current.score else current

        # 3) deterministic tie-break: lexical route name
        return candidate if candidate.route < current.route else current


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    r = Router(base / "config" / "routes.yaml")

    tests = [
        "What was the total revenue for Electronics in December 2024?",
        "Which region had the highest sales volume?",
        "What are the key features of the Wireless Bluetooth Headphones?",
        "How do customers rate the Air Fryer in terms of ease of cleaning?",
        "Which product has the best customer reviews and how does it perform in sales?",
        "Recommend a fitness product that is highly rated and sells well in the West region.",
    ]

    for q in tests:
        res = r.route(q)
        print(f"\nQ: {q}\n-> route={res.route} source={res.source} score={res.score} kw={res.matched_keywords} re={res.matched_regex} debug={res.debug}")