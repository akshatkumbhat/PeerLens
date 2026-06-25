"""Evaluation harness.

Runs the agent once per gold question and records τ-independent primitives
(agreement, what the consistent sample decided, and — for answerable questions —
whether that decision matches the gold institution+metric). The τ-sweep then
happens entirely in ``metrics.py`` over these records: the model is never
re-called per threshold.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from peerlens.agent.consistency import run_consistency
from peerlens.agent.model import ModelError, PlanModel
from peerlens.agent.resolve import resolve_institution
from peerlens.config import Settings, get_settings

GOLD_PATH = Path(__file__).with_name("gold.json")


@dataclass
class EvalRecord:
    id: str
    kind: str                       # "answerable" | "unanswerable"
    question: str
    agreement: float
    top_kind: str                   # "answer" | "abstain" | "invalid"
    predicted_reason: str | None    # abstain reason / "no_valid_plan" / None
    answer_correct: bool | None     # answerable + top_kind=="answer": matches gold?
    expected_reason: str | None     # unanswerable: the gold abstain reason


def load_gold(path: Path | None = None) -> dict:
    return json.loads((path or GOLD_PATH).read_text())


def _primitives(cr) -> tuple[str, str | None]:
    top_kind = cr.top_key[0]
    if top_kind == "abstain":
        return top_kind, cr.top_samples[0].abstention.reason.value
    if top_kind == "invalid":
        return top_kind, "no_valid_plan"
    return top_kind, None


def run_eval(
    con: duckdb.DuckDBPyConnection,
    model: PlanModel,
    settings: Settings | None = None,
    gold: dict | None = None,
    *,
    limit: int | None = None,
    pause: float = 0.0,
    on_error: str = "stop",  # "stop" -> return partial; "raise" -> propagate
) -> list[EvalRecord]:
    """Run the agent over the gold set; return one record per question.

    Resilient to API failures: on a ModelError, by default it returns the records
    collected so far (partial progress preserved) rather than losing everything.
    ``pause`` spaces questions to respect rate limits.
    """
    s = settings or get_settings()
    gold = gold or load_gold()
    records: list[EvalRecord] = []

    def _consistency(question: str):
        return run_consistency(
            con, model, question, n=s.agent_samples, temperature=s.agent_temperature
        )

    answerable = gold["answerable"][:limit] if limit else gold["answerable"]
    unanswerable = gold["unanswerable"][:limit] if limit else gold["unanswerable"]

    try:
        for i, case in enumerate(answerable):
            gold_ids = resolve_institution(con, case["institution"])
            gold_unitid = gold_ids[0] if len(gold_ids) == 1 else None
            cr = _consistency(case["question"])
            top_kind, reason = _primitives(cr)
            correct: bool | None = None
            if top_kind == "answer":
                rp = cr.top_samples[0].resolved
                correct = (
                    gold_unitid is not None
                    and rp.target_unitid == gold_unitid
                    and rp.metric == case["metric"]
                )
            records.append(
                EvalRecord(case["id"], "answerable", case["question"], cr.agreement,
                           top_kind, reason, correct, None)
            )
            if pause and i < len(answerable) - 1:
                time.sleep(pause)

        for i, case in enumerate(unanswerable):
            cr = _consistency(case["question"])
            top_kind, reason = _primitives(cr)
            records.append(
                EvalRecord(case["id"], "unanswerable", case["question"], cr.agreement,
                           top_kind, reason, None, case["expected_reason"])
            )
            if pause and i < len(unanswerable) - 1:
                time.sleep(pause)
    except ModelError as e:
        if on_error == "raise":
            raise
        print(f"  ! stopped early after {len(records)} question(s): {e}")

    return records


def save_records(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))


def load_records(path: Path) -> list[EvalRecord]:
    return [EvalRecord(**d) for d in json.loads(path.read_text())]
