# main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

# Load .env so LLM API keys are set (repo root .env, then cwd)
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
    load_dotenv()  # cwd as fallback
except ImportError:
    pass

from core.router import Router
from core.planner import Planner
from core.executor import Executor
from core.retriever import Retriever
from core.formatter import Formatter
from core.qa_engine import QAEngine


def _resolve_paths(project_root: Path) -> tuple[Path, Path]:
    """
    Resolve config file paths relative to the project root.
    """
    routes_yaml = project_root / "config" / "routes.yaml"
    settings_yaml = project_root / "config" / "settings.yaml"
    return routes_yaml, settings_yaml


def run_question(
    question: str,
    *,
    project_root: Optional[Path] = None,
    show_plan: bool = False,
    show_raw: bool = False,
) -> int:
    project_root = project_root or Path(__file__).resolve().parent
    routes_yaml, settings_yaml = _resolve_paths(project_root)

    router = Router(routes_yaml)
    planner = Planner(routes_yaml, settings_yaml)
    executor = Executor(settings_yaml)
    retriever = Retriever(settings_yaml)
    formatter = Formatter(settings_yaml)
    qa_engine = QAEngine(settings_yaml)

    route_result = router.route(question)

    plan = planner.build_plan(route_result.route, question, route_debug=route_result.debug)

    if show_plan:
        print("\n[ROUTER]")
        print(f"- route: {route_result.route}")
        print(f"- score: {route_result.score}")
        print(f"- matched_keywords: {route_result.matched_keywords}")
        print(f"- matched_regex: {route_result.matched_regex}")
        print("\n[PLAN]")
        for i, pc in enumerate(plan, 1):
            print(f"{i:02d}. [{pc.stage}] {pc.cmd}")

    results = executor.run_plan(plan)

    if show_raw:
        print("\n[RAW EXECUTION RESULTS]")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"stage: {r.stage}")
            print(f"cmd: {r.cmd}")
            print(f"returncode: {r.returncode} | duration: {r.duration_seconds:.2f}s | timed_out: {r.timed_out}")
            if r.stdout:
                print("\nSTDOUT:\n" + r.stdout)
            if r.stderr:
                print("\nSTDERR:\n" + r.stderr)

    blocks, summary = retriever.build_context(results)

    formatted = formatter.format(
        question=question,
        blocks=blocks,
        summary=summary,
        route=route_result.route,
    )

    response = qa_engine.answer(formatted)

    # Final output
    print(response.answer)

    # Optional footer for debugging/grade clarity (kept short)
    print("\n---")
    print(f"route: {route_result.route} | blocks_used: {formatted.meta.get('blocks_used')} | model: {response.model}")
    if "retrieval_summary" in formatted.meta:
        rs = formatted.meta["retrieval_summary"]
        print(f"retrieval: files={rs.get('unique_files')} stdout_chars={rs.get('total_stdout_chars')} stopped_early={rs.get('stopped_early')}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Bash-tools RAG for codebase Q&A")
    parser.add_argument("question", nargs="?", help="Question to ask about the target repo")
    parser.add_argument("--stdin", action="store_true", help="Read question from stdin")
    parser.add_argument("--show-plan", action="store_true", help="Print routing decision and command plan")
    parser.add_argument("--show-raw", action="store_true", help="Print raw command outputs (can be long)")

    args = parser.parse_args()

    if args.stdin:
        question = input().strip()
    else:
        question = (args.question or "").strip()

    if not question:
        print("Error: no question provided. Use: python main.py \"your question\"  or  python main.py --stdin")
        return 2

    return run_question(question, show_plan=args.show_plan, show_raw=args.show_raw)


if __name__ == "__main__":
    raise SystemExit(main())

    