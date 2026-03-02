# core/orchestrator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import yaml

from core.router import RouteResult


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class PlanStep:
    """
    One planned retrieval step.
    kind:
      - "csv"  : run CSV retriever
      - "text" : run Text retriever
      - "both" : convenience (expands to csv+text)
    """
    kind: str                 # "csv" | "text"
    top_k: Optional[int] = None
    notes: str = ""


@dataclass(frozen=True)
class OrchestrationPlan:
    route: str
    source: str               # "csv" | "text" | "both"
    steps: List[PlanStep]
    debug: Dict[str, Any]


# -----------------------------
# Orchestrator / Planner
# -----------------------------

class Orchestrator:
    """
    Builds an orchestration plan from RouteResult + YAML config.

    Part 2 expectation:
    - Router decides source: csv / text / both
    - Orchestrator decides which retrievers to run and with what parameters
    - Actual retrieval is done by csv_retriever.py and text_retriever.py
    """

    def __init__(self, routes_yaml_path: str | Path, settings_yaml_path: str | Path):
        self.routes_yaml_path = Path(routes_yaml_path)
        self.settings_yaml_path = Path(settings_yaml_path)

        self._routes_doc = self._load_yaml(self.routes_yaml_path)
        self._settings_doc = self._load_yaml(self.settings_yaml_path)

        self._routes: Dict[str, Any] = self._routes_doc.get("routes", {})
        if not self._routes:
            raise ValueError(f"No 'routes' found in {self.routes_yaml_path}")

        # Defaults from settings.yaml
        self._text_top_k_default = int(
            (((self._settings_doc.get("retrieval") or {}).get("text") or {}).get("top_k")) or 3
        )
        self._multi_text_top_k_default = int(
            (((self._settings_doc.get("retrieval") or {}).get("multi_source") or {}).get("text_top_k")) or 3
        )

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"YAML not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"YAML must parse to a dict, got: {type(doc)}")
        return doc

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().split())

    def build_plan(self, route_result: RouteResult, question: str) -> OrchestrationPlan:
        """
        Translate a RouteResult into concrete steps.

        Minimal but robust logic:
        - source == csv  -> [csv]
        - source == text -> [text]
        - source == both -> [csv, text] (order matters: compute facts first, then textual evidence)
        """
        q = self._normalize(question)
        route = route_result.route
        source = route_result.source

        route_cfg = self._routes.get(route, {}) or {}
        strategy = route_cfg.get("strategy", {}) or {}
        strategy_type = str(strategy.get("type", "")).strip()

        steps: List[PlanStep] = []
        debug: Dict[str, Any] = {
            "route": route,
            "source": source,
            "strategy_type": strategy_type,
        }

        if source == "csv":
            steps.append(PlanStep(kind="csv", notes="Structured computation from CSV"))
        elif source == "text":
            top_k = self._get_text_top_k(route_cfg, fallback=self._text_top_k_default)
            steps.append(PlanStep(kind="text", top_k=top_k, notes="Keyword retrieval from unstructured text"))
        elif source == "both":
            # Run CSV first so the LLM gets the numeric facts, then text evidence.
            steps.append(PlanStep(kind="csv", notes="Structured computation from CSV"))
            top_k = self._get_multi_text_top_k(route_cfg, fallback=self._multi_text_top_k_default)
            steps.append(PlanStep(kind="text", top_k=top_k, notes="Text evidence for features/reviews"))
        else:
            # Fallback to text
            top_k = self._text_top_k_default
            steps.append(PlanStep(kind="text", top_k=top_k, notes="Fallback to text retrieval"))

        # Optional heuristics: adjust top_k when the question asks for "best/recommend"
        if any(w in q.lower() for w in ["best", "recommend", "highly rated", "top product"]):
            for i, s in enumerate(steps):
                if s.kind == "text":
                    boosted = max(s.top_k or self._text_top_k_default, 5)
                    steps[i] = PlanStep(kind="text", top_k=boosted, notes=s.notes + " (boosted top_k for ranking task)")
                    debug["boosted_text_top_k"] = boosted

        return OrchestrationPlan(route=route, source=source, steps=steps, debug=debug)

    @staticmethod
    def _get_text_top_k(route_cfg: Dict[str, Any], fallback: int) -> int:
        strat = route_cfg.get("strategy", {}) or {}
        top_k = strat.get("top_k")
        try:
            return int(top_k) if top_k is not None else fallback
        except Exception:
            return fallback

    def _get_multi_text_top_k(self, route_cfg: Dict[str, Any], fallback: int) -> int:
        strat = route_cfg.get("strategy", {}) or {}
        top_k = strat.get("text_top_k")
        try:
            return int(top_k) if top_k is not None else fallback
        except Exception:
            return fallback


# -----------------------------
# Optional manual test
# -----------------------------
if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    from core.router import Router

    router = Router(base / "config" / "routes.yaml")
    orch = Orchestrator(base / "config" / "routes.yaml", base / "config" / "settings.yaml")

    questions = [
        "What was the total revenue for Electronics in December 2024?",
        "What are the key features of the Wireless Bluetooth Headphones?",
        "Which product has the best customer reviews and how does it perform in sales?",
        "Recommend a fitness product that is highly rated and sells well in the West region.",
    ]

    for q in questions:
        rr = router.route(q)
        plan = orch.build_plan(rr, q)
        print(f"\nQ: {q}\nroute={plan.route} source={plan.source} steps={plan.steps} debug={plan.debug}")