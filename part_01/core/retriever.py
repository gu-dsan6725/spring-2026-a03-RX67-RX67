# core/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re

import yaml

from core.executor import CommandResult


@dataclass(frozen=True)
class ContextBlock:
    """
    A normalized unit of context to pass to formatter/LLM.
    """
    route: str
    stage: str
    cmd: str
    stdout: str
    stderr: str
    duration_seconds: float
    returncode: int
    timed_out: bool
    truncated_stdout: bool
    truncated_stderr: bool
    referenced_files: List[str]


@dataclass(frozen=True)
class RetrievalSummary:
    """
    Useful for debugging / logging / grading.
    """
    total_blocks: int
    total_stdout_chars: int
    total_stderr_chars: int
    unique_files: int
    stopped_early: bool
    stop_reason: str


class Retriever:
    """
    Retriever merges executor outputs into a compact context bundle.

    Responsibilities:
    - Filter empty/noisy results
    - Extract referenced file paths (best-effort)
    - Enforce global context budgets (chars)
    - Early-stop based on settings (min files / min chars)
    - Return ordered ContextBlocks + a summary
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        exec_cfg = self._settings.get("execution", {}) or {}
        self.max_total_out = int(exec_cfg.get("max_total_output_chars", 20000))

        ret_cfg = self._settings.get("retrieval", {}) or {}
        self.stop_if_min_files = int(ret_cfg.get("stop_if_min_files_found", 2))
        self.stop_if_min_chars = int(ret_cfg.get("stop_if_min_total_chars", 1500))

        exclude_cfg = self._settings.get("exclude", {}) or {}
        self.exclude_dirs: List[str] = list(exclude_cfg.get("directories", []) or [])
        self.exclude_file_patterns: List[str] = list(exclude_cfg.get("file_patterns", []) or [])

        self._file_regexes = self._build_file_regexes()

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def _build_file_regexes(self) -> List[re.Pattern]:
        """
        Best-effort file path detection for typical outputs:
        - rg:  path:line:match
        - find: ./path or path
        - cat/sed: mentions file names
        """
        # 1) rg style: some/path/file.ext:123: ...
        rg_like = re.compile(r"(?P<path>[^:\n\r\t]+?\.[A-Za-z0-9]{1,8}):(?P<line>\d+):")
        # 2) general paths containing slash and extension, sometimes prefixed with ./ or not
        path_with_ext = re.compile(r"(?P<path>(?:\./)?[A-Za-z0-9_\-./]+?\.[A-Za-z0-9]{1,8})")
        # 3) Dockerfile / Makefile without extension
        special_files = re.compile(r"(?P<path>(?:\./)?(?:Dockerfile(?:\.[A-Za-z0-9_\-]+)?|Makefile))\b")
        # 4) YAML / TOML / JSON w/out slash maybe
        manifest_files = re.compile(r"(?P<path>(?:\./)?(?:pyproject\.toml|package\.json|poetry\.lock|pnpm-lock\.yaml|yarn\.lock|package-lock\.json|Pipfile|setup\.cfg|requirements[^ \n\r\t]*\.txt))\b")

        return [rg_like, special_files, manifest_files, path_with_ext]

    def build_context(
        self,
        results: List[CommandResult],
        *,
        allow_stderr_blocks: bool = True,
    ) -> Tuple[List[ContextBlock], RetrievalSummary]:
        """
        Convert command results into a compact list of ContextBlocks.
        Applies early stop when enough evidence is collected.
        """
        blocks: List[ContextBlock] = []
        seen_cmds: Set[str] = set()
        all_files: Set[str] = set()

        total_stdout = 0
        total_stderr = 0
        stopped_early = False
        stop_reason = ""

        for res in results:
            # Deduplicate exact commands (paranoid, usually planner already deduped)
            cmd_key = res.cmd.strip()
            if cmd_key in seen_cmds:
                continue
            seen_cmds.add(cmd_key)

            stdout = (res.stdout or "").strip("\n")
            stderr = (res.stderr or "").strip("\n")

            # Filter out purely empty outputs unless we want stderr-only blocks
            has_stdout = bool(stdout.strip())
            has_stderr = bool(stderr.strip())

            if not has_stdout and (not allow_stderr_blocks or not has_stderr):
                continue

            # Extract referenced files from stdout+stderr
            referenced_files = self._extract_files(stdout + "\n" + stderr)
            referenced_files = self._filter_excluded_files(referenced_files)
            for f in referenced_files:
                all_files.add(f)

            block = ContextBlock(
                route=res.route,
                stage=res.stage,
                cmd=res.cmd,
                stdout=stdout,
                stderr=stderr if allow_stderr_blocks else "",
                duration_seconds=res.duration_seconds,
                returncode=res.returncode,
                timed_out=res.timed_out,
                truncated_stdout=res.truncated_stdout,
                truncated_stderr=res.truncated_stderr,
                referenced_files=referenced_files,
            )

            blocks.append(block)

            total_stdout += len(stdout)
            total_stderr += len(stderr)

            # Hard budget stop (avoid blowing up context)
            if total_stdout >= self.max_total_out:
                stopped_early = True
                stop_reason = f"Reached max_total_output_chars={self.max_total_out}"
                break

            # Soft early stop: enough evidence to answer
            if self._should_stop_early(total_stdout, all_files):
                stopped_early = True
                stop_reason = (
                    f"Early stop: stdout_chars={total_stdout} (>= {self.stop_if_min_chars}) "
                    f"and unique_files={len(all_files)} (>= {self.stop_if_min_files})"
                )
                break

        summary = RetrievalSummary(
            total_blocks=len(blocks),
            total_stdout_chars=total_stdout,
            total_stderr_chars=total_stderr,
            unique_files=len(all_files),
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
        return blocks, summary

    def _should_stop_early(self, total_stdout_chars: int, all_files: Set[str]) -> bool:
        """
        Simple stop condition:
        - Have enough text AND enough distinct files
        """
        if total_stdout_chars >= self.stop_if_min_chars and len(all_files) >= self.stop_if_min_files:
            return True
        return False

    def _extract_files(self, text: str) -> List[str]:
        """
        Best-effort file extraction. Returns normalized paths (no leading ./).
        """
        if not text:
            return []
        found: List[str] = []
        for pat in self._file_regexes:
            for m in pat.finditer(text):
                p = m.groupdict().get("path")
                if not p:
                    continue
                p = p.strip()
                # Normalize
                if p.startswith("./"):
                    p = p[2:]
                # Remove trailing punctuation
                p = p.rstrip("),.;:\"'")
                # Very short / suspicious
                if len(p) < 3:
                    continue
                found.append(p)
        # Keep order, dedupe
        out: List[str] = []
        seen = set()
        for p in found:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _filter_excluded_files(self, files: List[str]) -> List[str]:
        """
        Remove paths that live under excluded directories or match excluded file patterns.
        """
        if not files:
            return files

        out: List[str] = []
        for p in files:
            # Excluded directory?
            if any(p == d or p.startswith(d.rstrip("/") + "/") for d in self.exclude_dirs):
                continue

            # Excluded file patterns (very lightweight glob-ish check)
            # We don't implement full fnmatch here; just do suffix/contains heuristics.
            lower = p.lower()
            blocked = False
            for pat in self.exclude_file_patterns:
                pat = pat.strip()
                if not pat:
                    continue
                # Common patterns like "*.png", "*.lock"
                if pat.startswith("*.") and lower.endswith(pat[1:].lower()):
                    blocked = True
                    break
                # Fallback: if pattern substring appears (rare)
                if pat.strip("*").lower() and pat.strip("*").lower() in lower:
                    blocked = True
                    break
            if blocked:
                continue

            out.append(p)

        return out


# Optional quick manual test:
if __name__ == "__main__":
    # This block doesn't execute bash; it's just a sanity check for parsing.
    from core.planner import PlannedCommand
    from core.executor import CommandResult

    dummy = [
        CommandResult(
            route="api_endpoints",
            stage="List endpoints (decorators)",
            cmd="rg -n \"router\\.(get|post)\\(\" -S . || true",
            returncode=0,
            duration_seconds=0.2,
            stdout="backend/api/routes.py:12:@router.get('/health')\nbackend/api/routes.py:20:@router.post('/token')\n",
            stderr="",
            truncated_stdout=False,
            truncated_stderr=False,
            timed_out=False,
        )
    ]

    base = Path(__file__).resolve().parents[1]
    r = Retriever(base / "config" / "settings.yaml")
    blocks, summary = r.build_context(dummy)
    print(summary)
    for b in blocks:
        print("files:", b.referenced_files)
        print("stdout:", b.stdout)