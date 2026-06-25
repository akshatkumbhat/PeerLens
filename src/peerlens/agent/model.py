"""Provider-swappable plan model.

A ``PlanModel`` turns a question + linked catalog into a ``QueryPlan`` (or None
if it can't produce a valid one). The pipeline samples it N times at nonzero
temperature for self-consistency. ``FakePlanModel`` makes the whole agent
testable offline; ``GeminiPlanModel`` is the default live provider (needs
``GEMINI_API_KEY``). Adding Ollama/Claude is one new class implementing the same
protocol — the pipeline never changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from peerlens.agent.catalog import LinkedCatalog, metric_menu
from peerlens.agent.plan import QueryPlan


class PlanModel(Protocol):
    def propose(
        self, question: str, linked: LinkedCatalog, *, temperature: float
    ) -> QueryPlan | None: ...


class ModelError(RuntimeError):
    pass


def build_prompt(question: str, linked: LinkedCatalog) -> str:
    """Schema-linked instruction: extract a grounded query plan as JSON."""
    metrics = ", ".join(m.key for m in linked.metrics)
    return (
        "You translate a question about U.S. higher-education data into a STRICT JSON "
        "query plan. Output JSON only, no prose.\n\n"
        "Schema (relevant slice):\n"
        f"  tables: {', '.join(linked.tables)}\n"
        f"  metrics you MAY use: {metrics}\n"
        f"  full metric menu: {metric_menu()}\n\n"
        "JSON shape:\n"
        '  {"intent": "single"|"compare",\n'
        '   "institution": "<name as written in the question>",\n'
        '   "metric": "<one metric key>",\n'
        '   "comparison": {"kind": "none"|"peers"|"aspirants"|"explicit",\n'
        '                  "institutions": ["<name>", ...], "k": 10},\n'
        '   "years": [2020]}\n\n'
        "Rules: pick exactly one metric key from the menu. Use intent=compare with "
        "comparison.kind=peers/aspirants for peer/aspirant questions, explicit when the "
        "question names the comparison institutions, else single with kind=none. Only "
        "year 2020 is available.\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


class FakePlanModel:
    """Deterministic model for tests: replays the configured plans in order."""

    def __init__(self, plans: Sequence[QueryPlan | None]):
        self._plans = list(plans)
        self._i = 0

    def propose(
        self, question: str, linked: LinkedCatalog, *, temperature: float
    ) -> QueryPlan | None:
        if not self._plans:
            return None
        plan = self._plans[min(self._i, len(self._plans) - 1)]
        self._i += 1
        return plan


class GeminiPlanModel:
    """Gemini-backed plan model via the REST API (httpx only, no SDK).

    Requests constrained JSON (``responseMimeType: application/json``) and parses
    it into a ``QueryPlan``; a malformed sample returns None (a dropped vote, not
    a crash), which is exactly what self-consistency expects.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", timeout: float = 30.0):
        if not api_key:
            raise ModelError("GEMINI_API_KEY is not set — cannot use the Gemini provider.")
        import httpx

        self._api_key = api_key
        self._model = model
        self._client = httpx.Client(timeout=timeout)

    def propose(
        self, question: str, linked: LinkedCatalog, *, temperature: float
    ) -> QueryPlan | None:
        import httpx

        url = f"{self._BASE}/{self._model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": build_prompt(question, linked)}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = self._client.post(
                url, headers={"x-goog-api-key": self._api_key}, json=body
            )
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (httpx.HTTPError, KeyError, IndexError):
            return None
        try:
            return QueryPlan.model_validate_json(text)
        except Exception:  # malformed sample — a dropped vote, not a crash
            return None


def get_plan_model(settings=None) -> PlanModel:
    """Factory: build the configured provider's plan model."""
    from peerlens.config import get_settings

    s = settings or get_settings()
    provider = s.peerlens_model_provider.lower()
    if provider == "gemini":
        return GeminiPlanModel(s.gemini_api_key, getattr(s, "gemini_model", "gemini-2.0-flash"))
    raise ModelError(
        f"provider '{provider}' is not wired yet; implement a PlanModel for it "
        "(the pipeline is provider-agnostic)."
    )
