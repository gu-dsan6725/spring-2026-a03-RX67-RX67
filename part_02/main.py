# main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from core.router import Router
from core.planner import Orchestrator
from core.csv_retriever import CSVRetriever, CSVResult
from core.text_retriever import TextRetriever, TextResult
from core.combiner import Combiner
from core.formatter import Formatter
from core.qa_engine import QAEngine


def _project_root() -> Path:
    # main.py lives at project root in the suggested structure
    return Path(__file__).resolve().parent


def _config_paths(root: Path) -> Tuple[Path, Path]:
    return root / "configs" / "routes.yaml", root / "configs" / "settings.yaml"


def run_question(
    question: str,
    *,
    show_route: bool = False,
    show_plan: bool = False,
    show_evidence: bool = False,
) -> int:
    root = _project_root()
    routes_yaml, settings_yaml = _config_paths(root)

    router = Router(routes_yaml)
    orch = Orchestrator(routes_yaml, settings_yaml)

    csv_ret = CSVRetriever(settings_yaml)
    text_ret = TextRetriever(settings_yaml)

    combiner = Combiner(
        max_chars=int(
            (((((yaml_load(settings_yaml).get("retrieval") or {}).get("multi_source") or {}).get("max_combined_chars"))
             or 6000))
        )
    )
    formatter = Formatter(settings_yaml, prompts_dir=root / "prompts")
    qa = QAEngine(settings_yaml)

    # 1) route
    route_result = router.route(question)
    if show_route:
        print("[ROUTER]")
        print(f"- route: {route_result.route}")
        print(f"- source: {route_result.source}")
        print(f"- score: {route_result.score}")
        print(f"- matched_keywords: {route_result.matched_keywords}")
        print(f"- matched_regex: {route_result.matched_regex}")
        print("")

    # 2) plan steps
    plan = orch.build_plan(route_result, question)
    if show_plan:
        print("[PLAN]")
        print(f"- route: {plan.route} | source: {plan.source}")
        for i, s in enumerate(plan.steps, 1):
            print(f"  {i}. kind={s.kind} top_k={s.top_k} notes={s.notes}")
        print("")

    # 3) retrieve
    csv_result: Optional[CSVResult] = None
    text_result: Optional[TextResult] = None

    for step in plan.steps:
        if step.kind == "csv":
            csv_result = csv_ret.run(question)
        elif step.kind == "text":
            text_result = text_ret.run(question, top_k=step.top_k)
        else:
            raise ValueError(f"Unknown plan step kind: {step.kind}")

    # 4) combine
    combined = combiner.combine(
        question=question,
        route=plan.route,
        source=plan.source,
        csv_result=csv_result,
        text_result=text_result,
    )

    # 5) format prompt
    formatted = formatter.format(combined)

    if show_evidence:
        print("[COMBINED EVIDENCE]")
        print(combined.evidence_text[:8000])
        print("\n---\n")

    # 6) ask LLM
    response = qa.answer(formatted)

    # 7) output
    print(response.answer)
    return 0


def yaml_load(settings_yaml: Path):
    # tiny helper to avoid adding a new dependency location
    import yaml
    with settings_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 2: Multi-Source RAG with Routing (CSV + Text)")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--stdin", action="store_true", help="Read question from stdin")
    parser.add_argument("--show-route", action="store_true", help="Print routing decision")
    parser.add_argument("--show-plan", action="store_true", help="Print orchestration plan")
    parser.add_argument("--show-evidence", action="store_true", help="Print combined evidence fed to the model")

    args = parser.parse_args()

    if args.stdin:
        question = input().strip()
    else:
        question = (args.question or "").strip()

    if not question:
        print("Error: No question provided.\n"
              "Usage: python main.py \"your question\"  or  python main.py --stdin")
        return 2

    return run_question(
        question,
        show_route=args.show_route,
        show_plan=args.show_plan,
        show_evidence=args.show_evidence,
    )


if __name__ == "__main__":
    raise SystemExit(main())