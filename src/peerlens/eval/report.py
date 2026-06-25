"""Render the eval report: a markdown summary, the risk-coverage plot, and a
README section update."""

from __future__ import annotations

from pathlib import Path

from peerlens.eval.harness import EvalRecord
from peerlens.eval.metrics import Metrics, metrics_at, operating_point, risk_coverage

README_START = "<!-- EVAL:START -->"
README_END = "<!-- EVAL:END -->"


def _pct(x: float) -> str:
    return f"{x:.1%}"


def render_markdown(records: list[EvalRecord], op: Metrics, default_tau: float) -> str:
    at_default = metrics_at(records, default_tau)
    n_ans = sum(1 for r in records if r.kind == "answerable")
    n_unans = sum(1 for r in records if r.kind == "unanswerable")
    lines = [
        f"**Eval set:** {len(records)} questions ({n_ans} answerable, {n_unans} unanswerable). "
        f"Each scored over self-consistency samples; the threshold τ is swept analytically.",
        "",
        f"**Operating point (selective risk ≤ 2%): τ = {op.tau:.2f}**",
        "",
        "| Metric | At τ = "
        f"{op.tau:.2f} (chosen) | At τ = {default_tau:.2f} (default) |",
        "|---|---|---|",
        f"| Coverage (answered) | {_pct(op.coverage)} | {_pct(at_default.coverage)} |",
        f"| **Confident-wrong rate** | **{_pct(op.confident_wrong_rate)}** | {_pct(at_default.confident_wrong_rate)} |",
        f"| Selective risk (error among answered) | {_pct(op.selective_risk)} | {_pct(at_default.selective_risk)} |",
        f"| Execution accuracy (EX) | {_pct(op.execution_accuracy)} | {_pct(at_default.execution_accuracy)} |",
        f"| Abstention recall | {_pct(op.abstention_recall)} | {_pct(at_default.abstention_recall)} |",
        f"| Over-abstention | {_pct(op.over_abstention)} | {_pct(at_default.over_abstention)} |",
        "",
        "![Risk-coverage curve](docs/eval/risk_coverage.png)",
    ]
    return "\n".join(lines)


def plot_risk_coverage(records: list[EvalRecord], op: Metrics, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = risk_coverage(records)
    xs = [m.coverage for m in curve]
    ys = [m.selective_risk for m in curve]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color="#4f46e5")
    ax.scatter([op.coverage], [op.selective_risk], color="#dc2626", zorder=5,
               label=f"operating point (τ={op.tau:.2f})")
    ax.axhline(0.02, ls="--", lw=1, color="#94a3b8", label="2% risk target")
    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Selective risk (error among answered)")
    ax.set_title("PeerLens risk-coverage")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(0.1, max(ys) + 0.02))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def update_readme(readme: Path, section_md: str) -> bool:
    """Replace the content between the EVAL markers; return True if updated."""
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    pre = text.split(README_START)[0]
    post = text.split(README_END)[1]
    readme.write_text(f"{pre}{README_START}\n{section_md}\n{README_END}{post}")
    return True


def write_report(records: list[EvalRecord], out_dir: Path, default_tau: float) -> Metrics:
    """Write report.md + risk_coverage.png into out_dir; return the operating point."""
    op = operating_point(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_risk_coverage(records, op, out_dir / "risk_coverage.png")
    (out_dir / "report.md").write_text(render_markdown(records, op, default_tau))
    return op
