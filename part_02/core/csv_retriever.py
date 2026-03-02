# core/csv_retriever.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import re

import pandas as pd
import yaml


@dataclass(frozen=True)
class CSVResult:
    """
    Normalized structured result to pass downstream (combiner/formatter/LLM).
    """
    query_type: str
    answer_text: str
    evidence: str
    metrics: Dict[str, Any]


class CSVRetriever:
    """
    CSV retriever for Part 2.

    It is intentionally designed to handle the 2 CSV-only test queries robustly:
      Q1: total revenue for <category> in <month/year>
      Q2: which region had the highest sales volume (units sold)

    For "both" routes, we still produce structured metrics (sales/revenue) that
    the LLM can cite.

    Implementation notes:
    - Uses pandas for correctness and simplicity
    - The router/orchestrator decides whether CSVRetriever is invoked
    """

    def __init__(self, settings_yaml_path: str | Path):
        self.settings_yaml_path = Path(settings_yaml_path)
        self._settings = self._load_yaml(self.settings_yaml_path)

        data_cfg = self._settings.get("data", {}) or {}
        csv_path = data_cfg.get("csv_path")
        if not csv_path:
            raise ValueError("settings.yaml missing data.csv_path")

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            # allow running from different cwd: resolve relative to project root
            self.csv_path = (self.settings_yaml_path.parent.parent / csv_path).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        csv_cfg = ((self._settings.get("retrieval") or {}).get("csv") or {})
        self.col_date = str(csv_cfg.get("date_column", "date"))
        self.col_category = str(csv_cfg.get("category_column", "category"))
        self.col_region = str(csv_cfg.get("region_column", "region"))
        self.col_units = str(csv_cfg.get("units_column", "units_sold"))
        self.col_revenue = str(csv_cfg.get("revenue_column", "total_revenue"))

        self._df: Optional[pd.DataFrame] = None

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"settings.yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"settings.yaml must parse to a dict, got: {type(doc)}")
        return doc

    def _load_df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        df = pd.read_csv(self.csv_path)
        # Basic normalization
        if self.col_date not in df.columns:
            raise ValueError(f"CSV missing date column '{self.col_date}'")
        df[self.col_date] = pd.to_datetime(df[self.col_date], errors="coerce")

        self._df = df
        return df

    # ---------------------------
    # Public API
    # ---------------------------

    def run(self, question: str) -> CSVResult:
        """
        Attempt to answer the question using structured data.

        If the question doesn't match known patterns, returns a generic summary
        or a "cannot answer" note with suggestions.
        """
        df = self._load_df()
        q = question.strip()

        # Try Q1-like: total revenue for category in month
        parsed = self._parse_revenue_by_category_month(q)
        if parsed is not None:
            category, year, month = parsed
            return self._answer_total_revenue_by_category_month(df, category, year, month)

        # Try Q2-like: top region by sales volume
        if self._looks_like_top_region_sales(q):
            return self._answer_top_region_by_units(df)

        # Fallback: return a small dataset summary (helpful for BOTH questions)
        return self._fallback_summary(df, q)

    # ---------------------------
    # Pattern parsing (minimal but stable)
    # ---------------------------

    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }

    def _parse_revenue_by_category_month(self, q: str) -> Optional[Tuple[str, int, int]]:
        """
        Extract (category, year, month) from queries like:
          - "total revenue for Electronics in December 2024"
          - "December 2024 Electronics total revenue"
        """
        q_low = q.lower()

        # Month + Year
        month = None
        for mname, mnum in self.MONTHS.items():
            if mname in q_low:
                month = mnum
                break
        if month is None:
            return None

        year_match = re.search(r"\b(20\d{2})\b", q_low)
        if not year_match:
            return None
        year = int(year_match.group(1))

        # Category: look for known categories from data or common ones
        # We'll try to infer by matching capitalized token sequences or known category words.
        # Safer approach: scan unique categories and pick one that appears in question.
        df = self._load_df()
        categories = sorted({str(x) for x in df[self.col_category].dropna().unique()})
        cat_hit = None
        for cat in categories:
            if cat.lower() in q_low:
                cat_hit = cat
                break

        # If no category found, return None (can't compute Q1)
        if not cat_hit:
            return None

        # Also require revenue-ish intent
        if "revenue" not in q_low:
            return None

        return cat_hit, year, month

    def _looks_like_top_region_sales(self, q: str) -> bool:
        q_low = q.lower()
        has_region = "region" in q_low
        has_sales = any(w in q_low for w in ["highest sales", "sales volume", "most sales", "top region", "units sold"])
        return has_region and has_sales

    # ---------------------------
    # Answer builders
    # ---------------------------

    def _answer_total_revenue_by_category_month(self, df: pd.DataFrame, category: str, year: int, month: int) -> CSVResult:
        mask = (
            (df[self.col_category].astype(str) == category)
            & (df[self.col_date].dt.year == year)
            & (df[self.col_date].dt.month == month)
        )
        sub = df.loc[mask]

        total = float(sub[self.col_revenue].sum()) if not sub.empty else 0.0
        n_rows = int(sub.shape[0])

        answer = f"Total revenue for category '{category}' in {year}-{month:02d} is {total:.2f}."
        evidence = (
            f"Computed by filtering {self.csv_path.name} where "
            f"{self.col_category}='{category}', year={year}, month={month}, "
            f"then summing column '{self.col_revenue}'. "
            f"Matched rows: {n_rows}."
        )

        metrics = {
            "category": category,
            "year": year,
            "month": month,
            "total_revenue": total,
            "matched_rows": n_rows,
            "csv_path": str(self.csv_path),
        }

        return CSVResult(
            query_type="total_revenue_by_category_month",
            answer_text=answer,
            evidence=evidence,
            metrics=metrics,
        )

    def _answer_top_region_by_units(self, df: pd.DataFrame) -> CSVResult:
        grp = df.groupby(self.col_region, dropna=False)[self.col_units].sum().sort_values(ascending=False)
        top_region = str(grp.index[0])
        top_units = float(grp.iloc[0])

        # Provide top-5 table as evidence
        top5 = grp.head(5)
        table_lines = ["region,units_sold_sum"]
        for r, v in top5.items():
            table_lines.append(f"{r},{float(v):.0f}")
        table = "\n".join(table_lines)

        answer = f"The region with the highest sales volume (units sold) is '{top_region}' with {top_units:.0f} units sold."
        evidence = (
            f"Grouped {self.csv_path.name} by '{self.col_region}', summed '{self.col_units}', and took the maximum.\n"
            f"Top-5 regions:\n{table}"
        )

        metrics = {
            "top_region": top_region,
            "top_units_sold": top_units,
            "top5_table": table,
            "csv_path": str(self.csv_path),
        }

        return CSVResult(
            query_type="top_region_by_units_sold",
            answer_text=answer,
            evidence=evidence,
            metrics=metrics,
        )

    def _fallback_summary(self, df: pd.DataFrame, q: str) -> CSVResult:
        """
        If the question doesn't match known structured patterns, provide a small summary
        that is still useful when combining sources (e.g., BOTH queries).
        """
        min_date = df[self.col_date].min()
        max_date = df[self.col_date].max()
        n = int(df.shape[0])

        categories = sorted({str(x) for x in df[self.col_category].dropna().unique()})
        regions = sorted({str(x) for x in df[self.col_region].dropna().unique()})

        answer = "The question does not match a supported structured query pattern. Returning dataset summary."
        evidence = (
            f"CSV summary from {self.csv_path.name}:\n"
            f"- rows: {n}\n"
            f"- date range: {min_date.date() if pd.notnull(min_date) else None} to {max_date.date() if pd.notnull(max_date) else None}\n"
            f"- categories: {', '.join(categories[:12])}{'...' if len(categories) > 12 else ''}\n"
            f"- regions: {', '.join(regions[:12])}{'...' if len(regions) > 12 else ''}\n"
        )

        metrics = {
            "rows": n,
            "min_date": str(min_date) if pd.notnull(min_date) else None,
            "max_date": str(max_date) if pd.notnull(max_date) else None,
            "num_categories": len(categories),
            "num_regions": len(regions),
            "csv_path": str(self.csv_path),
        }

        return CSVResult(
            query_type="fallback_summary",
            answer_text=answer,
            evidence=evidence,
            metrics=metrics,
        )


if __name__ == "__main__":
    # Minimal smoke test
    base = Path(__file__).resolve().parents[1]
    retriever = CSVRetriever(base / "config" / "settings.yaml")

    qs = [
        "What was the total revenue for Electronics in December 2024?",
        "Which region had the highest sales volume?",
        "Show me something about sales.",
    ]
    for q in qs:
        r = retriever.run(q)
        print("\nQ:", q)
        print(r.answer_text)
        print(r.evidence)