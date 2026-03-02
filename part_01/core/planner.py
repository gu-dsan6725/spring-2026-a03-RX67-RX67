# core/planner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import yaml


@dataclass(frozen=True)
class PlannedCommand:
    route: str
    stage: str
    cmd: str
    description: str = ""


class Planner:
    """
    Planner turns a routed query into a staged command plan.

    Inputs:
    - routes.yaml: per-route stages + command templates
    - settings.yaml: runtime caps (max stages/commands)

    Responsibilities:
    - Expand route -> stages -> commands into a flat list
    - Enforce caps (max_stages, max_commands_per_query)
    - Substitute ${QUERY} for general_search (sanitized tokens)
    - Provide a deterministic, debuggable plan
    """

    def __init__(
        self,
        routes_yaml_path: str | Path,
        settings_yaml_path: str | Path,
    ):
        self.routes_yaml_path = Path(routes_yaml_path)
        self.settings_yaml_path = Path(settings_yaml_path)

        self._routes_doc = self._load_yaml(self.routes_yaml_path)
        self._settings_doc = self._load_yaml(self.settings_yaml_path)

        self._routes: Dict[str, Any] = self._routes_doc.get("routes", {})
        if not self._routes:
            raise ValueError(f"No 'routes' found in {self.routes_yaml_path}")

        # Settings (with safe defaults)
        self._max_stages: int = int(self._settings_doc.get("retrieval", {}).get("max_stages", 4))
        self._max_cmds: int = int(self._settings_doc.get("execution", {}).get("max_commands_per_query", 20))

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

    @staticmethod
    def _sanitize_query_for_rg(query: str, max_terms: int = 6) -> str:
        """
        Convert the user's question into a reasonably safe search string
        to substitute into `${QUERY}` for general_search.

        Strategy:
        - lowercase
        - keep alphanumerics, underscore, dash; map others to space
        - remove very short tokens
        - take up to max_terms tokens
        - join with '|' to behave like an OR regex for rg (simple, effective)
        """
        q = query.lower()
        q = re.sub(r"[^a-z0-9_\-]+", " ", q)
        tokens = [t for t in q.split() if len(t) >= 3]
        tokens = tokens[:max_terms] if tokens else []
        if not tokens:
            return "TODO"  # executor will still run, but likely find nothing
        # Escape tokens for regex safety (paranoid but cheap)
        tokens = [re.escape(t) for t in tokens]
        return "|".join(tokens)

    def build_plan(
        self,
        route: str,
        question: str,
        *,
        route_debug: Optional[Dict[str, Any]] = None,
    ) -> List[PlannedCommand]:
        """
        Build a plan for a given route and question.

        route_debug is optional (e.g., matched keywords/regex), useful to set
        better defaults for general_search in the future. Not required now.
        """
        route = route.strip()
        question_norm = self._normalize(question)

        if route not in self._routes:
            # Unknown route -> fallback
            route = "general_search"

        route_cfg = self._routes.get(route, {})
        stages = route_cfg.get("stages", []) or []

        plan: List[PlannedCommand] = []
        stages_taken = 0

        for stage_cfg in stages:
            if stages_taken >= self._max_stages:
                break
            stage_name = str(stage_cfg.get("name", f"stage_{stages_taken+1}"))
            commands = stage_cfg.get("commands", []) or []

            for cmd in commands:
                if len(plan) >= self._max_cmds:
                    return self._dedupe_plan(plan)

                cmd_str = str(cmd)

                # Substitute ${QUERY} for general_search or any template usage
                if "${QUERY}" in cmd_str:
                    sub = self._sanitize_query_for_rg(question_norm)
                    cmd_str = cmd_str.replace("${QUERY}", sub)

                plan.append(
                    PlannedCommand(
                        route=route,
                        stage=stage_name,
                        cmd=cmd_str,
                        description=str(stage_cfg.get("description", "")),
                    )
                )

            stages_taken += 1

        # If the route had no stages/commands, fallback to a safe general search
        if not plan:
            sub = self._sanitize_query_for_rg(question_norm)
            plan = [
                PlannedCommand(
                    route="general_search",
                    stage="Broad keyword search (fallback)",
                    cmd=f"rg -n \"{sub}\" -S . || true",
                    description="Fallback search when no route stages are defined.",
                )
            ]

        return self._dedupe_plan(plan)

    @staticmethod
    def _dedupe_plan(plan: List[PlannedCommand]) -> List[PlannedCommand]:
        """
        Deduplicate identical commands while keeping order.
        This helps when routes have overlapping fallbacks.
        """
        seen = set()
        out: List[PlannedCommand] = []
        for pc in plan:
            key = pc.cmd.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(pc)
        return out


# Optional quick manual test:
if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    planner = Planner(base / "config" / "routes.yaml", base / "config" / "settings.yaml")

    tests: List[Tuple[str, str]] = [
        ("deps", "What Python dependencies does this project use?"),
        ("entrypoint", "What is the main entry point file for the registry service?"),
        ("repo_types", "What languages/file types are used in this repository?"),
        ("auth_flow", "How does the authentication/authorization flow work?"),
        ("api_endpoints", "List all API endpoints and the required scopes."),
        ("add_oauth_provider", "How do I add support for a new OAuth provider like Okta?"),
        ("general_search", "Where is the README?"),
    ]

    for route, q in tests:
        print(f"\n=== route={route} q={q}")
        plan = planner.build_plan(route, q)
        for i, pc in enumerate(plan, 1):
            print(f"{i:02d}. [{pc.stage}] {pc.cmd}")