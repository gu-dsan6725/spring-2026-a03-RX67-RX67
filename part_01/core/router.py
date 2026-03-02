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
    score: int
    matched_keywords: List[str]
    matched_regex: List[str]
    debug: Dict[str, Any]


class Router:
    """
    Rule-based query router driven by config/routes.yaml.

    Design goals:
    - Deterministic and debuggable
    - Uses regex (strong signal) + keywords (weak signal)
    - Break ties by priority, then score
    - Falls back to general_search
    """

    def __init__(self, routes_yaml_path: str | Path):
        self.routes_yaml_path = Path(routes_yaml_path)
        self._routes_doc = self._load_yaml(self.routes_yaml_path)
        self._routes: Dict[str, Dict[str, Any]] = self._routes_doc.get("routes", {})
        if not self._routes:
            raise ValueError(f"No 'routes' found in {self.routes_yaml_path}")

        # Precompile regex patterns per route for speed + early validation
        self._compiled_regex: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for route_name, cfg in self._routes.items():
            patterns = cfg.get("regex", []) or []
            compiled: List[Tuple[str, re.Pattern]] = []
            for p in patterns:
                try:
                    compiled.append((p, re.compile(p)))
                except re.error as e:
                    raise ValueError(
                        f"Invalid regex for route '{route_name}': {p!r} ({e})"
                    ) from e
            self._compiled_regex[route_name] = compiled

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
            if not kw_norm:
                continue
            if kw_norm in q_norm:
                hits.append(kw)
        return hits

    def route(self, question: str) -> RouteResult:
        q_norm = self._normalize(question)

        # Scoring: regex matches are strong signals; keywords are weaker.
        REGEX_HIT = 6
        KEYWORD_HIT = 2

        best: Optional[RouteResult] = None

        for route_name, cfg in self._routes.items():
            if route_name == "general_search":
                continue  # only use as fallback

            priority = int(cfg.get("priority", 0))

            # Regex matches
            matched_regex: List[str] = []
            for raw, pat in self._compiled_regex.get(route_name, []):
                if pat.search(question) or pat.search(q_norm):
                    matched_regex.append(raw)

            # Keyword matches
            keywords = cfg.get("keywords", []) or []
            matched_keywords = self._match_keywords(q_norm, keywords)

            # Compute score
            score = len(matched_regex) * REGEX_HIT + len(matched_keywords) * KEYWORD_HIT

            # Optional: route-specific extra heuristics (small, targeted)
            score += self._bonus_heuristics(route_name, q_norm)

            candidate = RouteResult(
                route=route_name,
                score=score,
                matched_keywords=matched_keywords,
                matched_regex=matched_regex,
                debug={"priority": priority},
            )

            # Keep only meaningful candidates; otherwise everything with 0 would tie.
            if score <= 0:
                continue

            best = self._better(best, candidate)

        # Fallback if nothing matched
        if best is None:
            return RouteResult(
                route="general_search",
                score=0,
                matched_keywords=[],
                matched_regex=[],
                debug={"priority": int(self._routes.get("general_search", {}).get("priority", 0))},
            )

        return best

    def _better(self, current: Optional[RouteResult], candidate: RouteResult) -> RouteResult:
        """
        Compare two RouteResults. Choose the better one by:
        1) higher priority
        2) higher score
        3) deterministic tie-breaker: lexical route name
        """
        if current is None:
            return candidate

        cur_pri = int(current.debug.get("priority", 0))
        cand_pri = int(candidate.debug.get("priority", 0))

        if cand_pri != cur_pri:
            return candidate if cand_pri > cur_pri else current

        if candidate.score != current.score:
            return candidate if candidate.score > current.score else current

        # Deterministic tie-break
        return candidate if candidate.route < current.route else current

    @staticmethod
    def _bonus_heuristics(route_name: str, q_norm: str) -> int:
        """
        Small route-specific nudges to reduce common confusions.
        Keep this conservative—do not overfit.
        """
        bonus = 0

        if route_name == "api_endpoints":
            # If they explicitly ask to "list endpoints/routes/paths", that's strong.
            if any(x in q_norm for x in ["list endpoints", "api endpoints", "what routes", "routes are", "paths are"]):
                bonus += 3
            # Mention of scopes often co-occurs with endpoints questions.
            if "scope" in q_norm or "scopes" in q_norm:
                bonus += 2

        if route_name == "auth_flow":
            # If they ask about "flow" or "authorization flow", treat as strong.
            if "auth flow" in q_norm or "authentication flow" in q_norm or "authorization flow" in q_norm:
                bonus += 4
            # Token/jwt/bearer are good indicators for auth flow.
            if any(x in q_norm for x in ["jwt", "bearer", "token validation", "validate token", "decode token"]):
                bonus += 2

        if route_name == "add_oauth_provider":
            # "add support", "new provider", or a concrete provider name is very specific.
            if any(x in q_norm for x in ["add oauth", "new oauth", "add provider", "new provider", "okta", "oidc", "openid"]):
                bonus += 5

        if route_name == "deps":
            if any(x in q_norm for x in ["pyproject", "requirements", "package.json", "dependencies", "devdependencies"]):
                bonus += 2

        if route_name == "entrypoint":
            if any(x in q_norm for x in ["entry point", "entrypoint", "startup", "how to run", "uvicorn", "dockerfile", "docker-compose"]):
                bonus += 2

        if route_name == "repo_types":
            if any(x in q_norm for x in ["file types", "extensions", "languages used", "what languages"]):
                bonus += 2

        return bonus


# Optional quick manual test:
if __name__ == "__main__":
    r = Router(Path(__file__).resolve().parents[1] / "config" / "routes.yaml")
    tests = [
        "What Python dependencies does this project use?",
        "What is the main entry point file for the registry service?",
        "What languages/file types are used in this repository?",
        "How does the authentication/authorization flow work?",
        "List all API endpoints and the required scopes.",
        "How do I add support for a new OAuth provider like Okta?",
        "Where is the README?",
    ]
    for q in tests:
        res = r.route(q)
        print(f"\nQ: {q}\n-> route={res.route} score={res.score} kw={res.matched_keywords} re={res.matched_regex} debug={res.debug}")