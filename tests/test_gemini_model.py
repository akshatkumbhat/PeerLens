"""Offline tests for GeminiPlanModel: API errors raise, bad content drops."""

from __future__ import annotations

import httpx
import pytest

from peerlens.agent.catalog import link
from peerlens.agent.model import GeminiPlanModel, ModelError, build_prompt


def _model(handler, *, max_retries: int = 0) -> GeminiPlanModel:
    return GeminiPlanModel(
        api_key="dummy",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=max_retries,
        sleep=lambda _s: None,  # never actually sleep in tests
    )


def _ok_json(plan_json: str):
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": plan_json}]}}]}
        )

    return handler


def test_prompt_routes_out_of_scope_to_resolver() -> None:
    """Regression guard: the prompt must not launder out-of-scope requests into an
    in-menu metric or a default year — that's what made the agent confidently wrong.
    It should instead echo the real measure/year so the resolver can abstain."""
    p = build_prompt("graduation rate at X in 2021", link("retention")).lower()
    # the old laundering instructions are gone
    assert "pick exactly one metric key from the menu" not in p
    assert "only year 2020 is available" not in p
    # and the model is told to name the real measure/year instead
    assert "never substitute" in p
    assert "exact year" in p


def test_valid_json_parses_to_plan() -> None:
    plan = '{"intent":"single","institution":"Alpha University","metric":"retention_rate","years":[2020]}'
    out = _model(_ok_json(plan)).propose("q", link("retention"), temperature=0.0)
    assert out is not None
    assert out.institution == "Alpha University"


def test_quota_429_raises_model_error() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(ModelError, match="429"):
        _model(handler).propose("q", link("retention"), temperature=0.0)


def test_garbage_content_is_dropped_vote() -> None:
    out = _model(_ok_json("not json at all")).propose("q", link("retention"), temperature=0.0)
    assert out is None  # unparseable -> dropped vote, not an error


def test_empty_candidates_is_dropped_vote() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": []})  # e.g. safety block

    assert _model(handler).propose("q", link("retention"), temperature=0.0) is None


def test_429_then_success_retries() -> None:
    plan = '{"intent":"single","institution":"Alpha University","metric":"admit_rate","years":[2020]}'
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "retry in 1s"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": plan}]}}]})

    out = _model(handler, max_retries=2).propose("q", link("admit"), temperature=0.0)
    assert out is not None and out.metric == "admit_rate"
    assert calls["n"] == 2  # one retry after the 429


_PLAN = '{"intent":"single","institution":"Alpha University","metric":"admit_rate","years":[2020]}'


def _ok_resp() -> httpx.Response:
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": _PLAN}]}}]})


def test_multiple_keys_round_robin() -> None:
    """Comma-separated keys are spread across successive calls."""
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers["x-goog-api-key"])
        return _ok_resp()

    m = GeminiPlanModel(
        api_key="kA,kB",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _s: None,
    )
    for _ in range(4):
        m.propose("q", link("admit"), temperature=0.0)
    assert seen == ["kA", "kB", "kA", "kB"]


def test_429_rotates_to_next_key_without_waiting() -> None:
    """A throttled key hands off to a free one instead of sleeping out the limit."""
    seen: list[str] = []
    slept: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        key = req.headers["x-goog-api-key"]
        seen.append(key)
        if key == "kA":  # first key is rate-limited; second is fine
            return httpx.Response(429, json={"error": {"message": "retry in 30s"}})
        return _ok_resp()

    m = GeminiPlanModel(
        api_key="kA,kB",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=2,
        sleep=lambda s: slept.append(s),
    )
    out = m.propose("q", link("admit"), temperature=0.0)
    assert out is not None and out.metric == "admit_rate"
    assert seen == ["kA", "kB"]  # rotated to the second key
    assert slept == []  # never had to wait — the other key was free
