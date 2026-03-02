# core/formatter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pathlib import Path

from core.retriever import ContextBlock, RetrievalSummary


@dataclass(frozen=True)
class FormattedPrompt:
    """
    The final payload to send to the LLM.
    """
    prompt: str
    meta: Dict[str, Any]


class Formatter:
    """
    Formats ContextBlocks into a structured prompt for the LLM.

    Goals:
    - Make tool outputs easy to read
    - Preserve command provenance (stage + command)
    - Encourage citations by surfacing referenced_files
    - Keep within reasonable context limits
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        fmt_cfg = self._settings.get("formatting", {}) or {}
        self.include_command_labels = bool(fmt_cfg.get("include_command_labels", True))
        self.include_stage_labels = bool(fmt_cfg.get("include_stage_labels", True))
        self.show_file_paths = bool(fmt_cfg.get("show_file_paths", True))
        self.show_line_numbers_if_available = bool(fmt_cfg.get("show_line_numbers_if_available", True))
        self.max_context_blocks = int(fmt_cfg.get("max_context_blocks", 12))

        exec_cfg = self._settings.get("execution", {}) or {}
        self.max_total_output_chars = int(exec_cfg.get("max_total_output_chars", 20000))

        llm_cfg = self._settings.get("llm", {}) or {}
        self.require_citations = bool(llm_cfg.get("require_citations", True))
        self.forbid_hallucination = bool(llm_cfg.get("forbid_hallucination", True))

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def format(
        self,
        question: str,
        blocks: List[ContextBlock],
        summary: Optional[RetrievalSummary] = None,
        *,
        route: Optional[str] = None,
    ) -> FormattedPrompt:
        """
        Produce a prompt that includes:
        - System instructions for grounded answering
        - The question
        - Tool outputs grouped by stage/command
        - A short retrieval summary (optional)
        """
        blocks = blocks[: self.max_context_blocks]

        instruction_lines = [
            "You are a codebase question-answering assistant.",
            "Answer using ONLY the provided repository evidence.",
        ]
        if self.forbid_hallucination:
            instruction_lines.append("If the evidence is insufficient, say so explicitly and suggest what to search next.")
        if self.require_citations:
            instruction_lines.append("Cite file paths for every key claim. Prefer path:line when line numbers exist (e.g., file.py:12).")

        header = "\n".join(instruction_lines)

        meta: Dict[str, Any] = {
            "route": route or (blocks[0].route if blocks else None),
            "blocks_used": len(blocks),
            "max_context_blocks": self.max_context_blocks,
        }
        if summary is not None:
            meta["retrieval_summary"] = {
                "total_blocks": summary.total_blocks,
                "total_stdout_chars": summary.total_stdout_chars,
                "unique_files": summary.unique_files,
                "stopped_early": summary.stopped_early,
                "stop_reason": summary.stop_reason,
            }

        context_text = self._format_blocks(blocks)

        retrieval_info = ""
        if summary is not None:
            retrieval_info = (
                "\n\n[Retrieval Summary]\n"
                f"- blocks: {summary.total_blocks}\n"
                f"- used: {len(blocks)}\n"
                f"- stdout_chars: {summary.total_stdout_chars}\n"
                f"- unique_files: {summary.unique_files}\n"
                f"- stopped_early: {summary.stopped_early}\n"
                f"- stop_reason: {summary.stop_reason}\n"
            )

        prompt = (
            f"{header}\n\n"
            f"[Question]\n{question.strip()}\n"
            f"{retrieval_info}\n"
            f"[Evidence]\n{context_text}\n\n"
            f"[Answer Requirements]\n"
            f"- Provide a clear, direct answer.\n"
            f"- Include citations (file paths; prefer path:line).\n"
            f"- If multiple plausible interpretations exist, explain and cite evidence.\n"
        )

        return FormattedPrompt(prompt=prompt, meta=meta)

    def _format_blocks(self, blocks: List[ContextBlock]) -> str:
        """
        Render context blocks as readable sections.

        Format:
        === Block i (route, stage) ===
        Command: ...
        Files: ...
        STDOUT:
        ```text
        ...
        ```
        STDERR: (if any)
        """
        rendered: List[str] = []
        total_chars = 0

        for i, b in enumerate(blocks, 1):
            parts: List[str] = []
            title = f"=== Evidence Block {i} ==="
            parts.append(title)

            if self.include_stage_labels:
                parts.append(f"Route: {b.route}")
                parts.append(f"Stage: {b.stage}")

            if self.include_command_labels:
                parts.append(f"Command: {b.cmd}")

            if self.show_file_paths and b.referenced_files:
                # Keep it compact; too many files can be noisy
                files = b.referenced_files[:20]
                more = "" if len(b.referenced_files) <= 20 else f" (+{len(b.referenced_files) - 20} more)"
                parts.append(f"Referenced files: {', '.join(files)}{more}")

            # Add execution metadata (helpful for debugging but concise)
            parts.append(
                f"Return code: {b.returncode} | Duration: {b.duration_seconds:.2f}s"
                + (" | TIMED OUT" if b.timed_out else "")
                + (" | STDOUT TRUNCATED" if b.truncated_stdout else "")
                + (" | STDERR TRUNCATED" if b.truncated_stderr else "")
            )

            # Stdout
            if b.stdout.strip():
                parts.append("STDOUT:")
                parts.append("```text")
                parts.append(b.stdout.rstrip())
                parts.append("```")

            # Stderr (only if present)
            if b.stderr.strip():
                parts.append("STDERR:")
                parts.append("```text")
                parts.append(b.stderr.rstrip())
                parts.append("```")

            block_text = "\n".join(parts).strip() + "\n"
            rendered.append(block_text)

            total_chars += len(block_text)
            if total_chars >= self.max_total_output_chars:
                rendered.append("\n...[formatter stopped: reached max_total_output_chars]...\n")
                break

        return "\n".join(rendered).strip()


# Optional quick manual test:
if __name__ == "__main__":
    from core.executor import CommandResult
    from core.retriever import Retriever

    base = Path(__file__).resolve().parents[1]
    settings = base / "config" / "settings.yaml"

    retriever = Retriever(settings)
    formatter = Formatter(settings)

    dummy_results = [
        CommandResult(
            route="deps",
            stage="Extract Python dependencies (if present)",
            cmd="rg -n \"dependencies\" pyproject.toml",
            returncode=0,
            duration_seconds=0.1,
            stdout="pyproject.toml:12:dependencies = [\"fastapi\", \"uvicorn\"]\n",
            stderr="",
            truncated_stdout=False,
            truncated_stderr=False,
            timed_out=False,
        )
    ]

    blocks, summary = retriever.build_context(dummy_results)
    fp = formatter.format("What dependencies does this project use?", blocks, summary)
    print(fp.prompt[:1200])