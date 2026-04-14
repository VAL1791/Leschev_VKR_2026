from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dashboard.dashboard_core import ablation_summary, ablation_takeaways, benchmark_takeaways, significance_takeaways


def build_markdown(results: pd.DataFrame, feature_importance: pd.DataFrame, significance_tests: pd.DataFrame) -> str:
    core_results = results.copy()
    if "run_status" in core_results.columns:
        statuses = core_results["run_status"].fillna("done").astype(str).str.strip().str.lower()
        core_results = core_results[statuses == "done"].copy()

    lines: list[str] = []
    lines.append("# Defense Summary")
    lines.append("")
    lines.append("## Core benchmark")
    for line in benchmark_takeaways(core_results, feature_importance=feature_importance):
        lines.append(f"- {line}")

    lines.append("")
    lines.append("## Ablation reading")
    for line in ablation_takeaways(results):
        lines.append(f"- {line}")

    lines.append("")
    lines.append("## Significance reading")
    for line in significance_takeaways(significance_tests):
        lines.append(f"- {line}")

    summary = ablation_summary(results)
    if not summary.empty:
        lines.append("")
        lines.append("## Ablation table (NDCG@10, exc)")
        lines.append("")
        lines.append("| Dataset | Hybrid | No Content | No Collaborative | Delta vs No Content | Delta vs No Collaborative |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in summary.itertuples(index=False):
            lines.append(
                "| "
                + f"{row.dataset} | {row.HybridFeatureRerank:.4f} | {row.Hybrid_NoContentAblation:.4f} | "
                + f"{row.Hybrid_NoCollaborativeAblation:.4f} | {row.delta_vs_no_content:+.4f} | {row.delta_vs_no_collab:+.4f} |"
            )

    lines.append("")
    lines.append("## Final framing")
    lines.append("- Main line of the thesis: strong baselines -> hybrid contribution -> robustness/business interpretation.")
    lines.append("- MovieLens should be framed as a dense collaborative benchmark where EASE is already very hard to beat.")
    lines.append("- Google Local should be framed as the most convincing applied hybrid case: collaborative backbone plus profile-alignment features.")
    lines.append("- Goodreads should be framed as a content-dominant domain where hybrid helps, but the ablation shows that content signals currently carry the main gain.")
    lines.append("- Sequential models should stay phase-2 scope: one selective reference model is enough for completeness.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a defense-ready markdown summary from dashboard artifacts.")
    parser.add_argument("--artifacts-dir", default="dashboard/artifacts")
    parser.add_argument("--output", default="docs/DEFENSE_SUMMARY.md")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = Path(args.artifacts_dir)
    output_path = Path(args.output)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (project_root / artifacts_dir).resolve()
    if not output_path.is_absolute():
        output_path = (project_root / output_path).resolve()

    results = pd.read_csv(artifacts_dir / "results.csv")
    feature_importance = pd.read_csv(artifacts_dir / "feature_importance.csv")
    significance_tests = pd.read_csv(artifacts_dir / "significance_tests.csv")
    markdown = build_markdown(results, feature_importance, significance_tests)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[export] defense summary -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
