# core/executor.py
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.planner import PlannedCommand


@dataclass(frozen=True)
class CommandResult:
    route: str
    stage: str
    cmd: str

    returncode: int
    duration_seconds: float

    stdout: str
    stderr: str

    truncated_stdout: bool
    truncated_stderr: bool
    timed_out: bool


class Executor:
    """
    Executes PlannedCommand(s) against a target repository.

    Key features:
    - cwd = target repo
    - timeout per command
    - output truncation
    - optional rg excludes injection via --glob '!dir/**'
    - robust for pipe commands by running through bash -lc
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        self.target_repo = Path(self._settings.get("target_repo", ".")).resolve()

        exec_cfg = self._settings.get("execution", {}) or {}
        self.default_timeout = int(exec_cfg.get("default_timeout_seconds", 8))
        self.max_out = int(exec_cfg.get("max_output_chars_per_command", 8000))
        self.max_total_out = int(exec_cfg.get("max_total_output_chars", 20000))

        exclude_cfg = self._settings.get("exclude", {}) or {}
        self.exclude_dirs: List[str] = list(exclude_cfg.get("directories", []) or [])
        self.exclude_file_patterns: List[str] = list(exclude_cfg.get("file_patterns", []) or [])

        if not self.target_repo.exists():
            raise FileNotFoundError(f"target_repo does not exist: {self.target_repo}")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def run_plan(self, plan: List[PlannedCommand]) -> List[CommandResult]:
        """
        Run a list of PlannedCommand and return per-command results.

        Note: This does NOT implement early stop logic. Put early-stop in retriever
        (based on total collected chars, files found, etc).
        """
        results: List[CommandResult] = []
        total_chars = 0

        for pc in plan:
            if total_chars >= self.max_total_out:
                # Stop executing more commands if we've already collected too much.
                break

            res = self.run_one(pc)
            results.append(res)

            # Only count stdout toward budget (simpler). You can count stderr too if you want.
            total_chars += len(res.stdout)

        return results

    def run_one(self, pc: PlannedCommand) -> CommandResult:
        cmd = pc.cmd.strip()
        cmd = self._maybe_inject_rg_excludes(cmd)

        start = time.time()
        timed_out = False

        try:
            # Use bash -lc so pipes, redirects, &&, || all behave as expected.
            proc = subprocess.run(
                ["bash", "-lc", cmd],
                cwd=str(self.target_repo),
                capture_output=True,
                text=True,
                timeout=self.default_timeout,
            )
            returncode = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

        except subprocess.TimeoutExpired as e:
            timed_out = True
            returncode = 124  # common timeout code
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            # Add a small message for visibility
            stderr = (stderr + "\n[executor] command timed out").strip() + "\n"

        duration = time.time() - start

        stdout, trunc_out = self._truncate(stdout, self.max_out)
        stderr, trunc_err = self._truncate(stderr, self.max_out)

        return CommandResult(
            route=pc.route,
            stage=pc.stage,
            cmd=cmd,
            returncode=returncode,
            duration_seconds=duration,
            stdout=stdout,
            stderr=stderr,
            truncated_stdout=trunc_out,
            truncated_stderr=trunc_err,
            timed_out=timed_out,
        )

    @staticmethod
    def _truncate(s: str, max_chars: int) -> tuple[str, bool]:
        if s is None:
            return "", False
        if len(s) <= max_chars:
            return s, False
        head = s[: max_chars - 200]
        tail = s[-200:]
        return f"{head}\n...[truncated {len(s) - max_chars} chars]...\n{tail}", True

    def _maybe_inject_rg_excludes(self, cmd: str) -> str:
        """
        If a command appears to use ripgrep (rg), inject exclusion globs:
          --glob '!node_modules/**' etc.
        and optionally file pattern excludes like '!*.png' etc.

        This is best-effort and intentionally conservative:
        - If '--glob' already present, we still append ours (OK in rg).
        - If command doesn't contain 'rg ', we do nothing.
        - Works for most simple commands and many piped commands.
        """
        if "rg " not in cmd and not cmd.startswith("rg"):
            return cmd

        # If the command is something like: rg -n "x" -S . || true
        # we inject globs right after 'rg' token.
        globs: List[str] = []
        for d in self.exclude_dirs:
            d = d.strip().strip("/")
            if d:
                globs.append(f"--glob '!{d}/**'")

        for pat in self.exclude_file_patterns:
            pat = pat.strip()
            if pat:
                globs.append(f"--glob '!{pat}'")

        if not globs:
            return cmd

        # Try to inject after the first 'rg' occurrence.
        # Handle cases:
        # - "rg -n ..." (starts with rg)
        # - "test -d src && rg -n ..." (rg appears later)
        idx = cmd.find("rg")
        if idx == -1:
            return cmd

        # Ensure we are injecting at a token boundary
        # (very lightweight: only inject when 'rg' is followed by space or end)
        after = cmd[idx + 2 : idx + 3]
        if after not in ("", " "):
            return cmd

        injection = "rg " + " ".join(globs) + " "
        # Replace only the first token "rg " with "rg <globs> "
        # If command begins with "rg", easy; otherwise replace first "rg " occurrence.
        if cmd.startswith("rg "):
            return injection + cmd[len("rg ") :]
        return cmd[:idx] + injection + cmd[idx + len("rg ") :]

    # Optional helper if you want to validate target repo quickly
    def ping(self) -> bool:
        try:
            subprocess.run(
                ["bash", "-lc", "pwd && ls"],
                cwd=str(self.target_repo),
                capture_output=True,
                text=True,
                timeout=3,
            )
            return True
        except Exception:
            return False


# Optional quick manual test:
if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    exe = Executor(base / "config" / "settings.yaml")

    # Minimal smoke test without depending on your planner:
    sample = PlannedCommand(route="general_search", stage="smoke", cmd="ls -la | head")
    res = exe.run_one(sample)
    print("returncode:", res.returncode)
    print("stdout:\n", res.stdout)
    print("stderr:\n", res.stderr)