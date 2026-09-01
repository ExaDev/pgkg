"""Tests for the claude_code provider — all offline, no real claude CLI invoked."""
from __future__ import annotations

import asyncio
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PROPS_JSON = json.dumps({
    "propositions": [
        {
            "text": "Alice is a scientist.",
            "subject": "Alice",
            "predicate": "is",
            "object": "scientist",
            "object_is_literal": False,
        }
    ]
})


def _make_fake_sdk(response_text: str = _VALID_PROPS_JSON):
    """Build a fake claude_agent_sdk module that yields one AssistantMessage."""

    fake_sdk = types.ModuleType("claude_agent_sdk")

    class FakeTextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeAssistantMessage:
        def __init__(self, text: str) -> None:
            self.content = [FakeTextBlock(text)]

    class FakeClaudeAgentOptions:
        def __init__(self, model: str = "", system_prompt: str = "") -> None:
            self.model = model
            self.system_prompt = system_prompt

    async def fake_query(prompt, *, options=None):
        yield FakeAssistantMessage(response_text)

    fake_sdk.query = fake_query
    fake_sdk.ClaudeAgentOptions = FakeClaudeAgentOptions
    return fake_sdk


# ---------------------------------------------------------------------------
# test_claude_code_extract_returns_propositions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_code_extract_returns_propositions(monkeypatch):
    """_extract_claude_code returns parsed propositions from the SDK response."""
    fake_sdk = _make_fake_sdk(_VALID_PROPS_JSON)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    from pgkg.ml import _extract_claude_code

    props = await _extract_claude_code(
        "Alice is a scientist who works at CERN.",
        extractor_model="claude-haiku-4-5-20251001",
        max_propositions=20,
        system_prompt="You are a knowledge extraction assistant.",
    )

    assert len(props) == 1
    assert props[0].subject == "Alice"
    assert props[0].predicate == "is"
    assert props[0].object == "scientist"


# ---------------------------------------------------------------------------
# test_claude_code_handles_sdk_missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_code_handles_sdk_missing(monkeypatch):
    """_extract_claude_code raises RuntimeError with install hint when SDK is missing."""
    # Remove the module from sys.modules so import raises ImportError
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)  # type: ignore[arg-type]

    from pgkg.ml import _extract_claude_code

    with pytest.raises(RuntimeError, match="uv sync --extra claude_agent"):
        await _extract_claude_code(
            "some text",
            extractor_model="claude-haiku-4-5-20251001",
            max_propositions=20,
            system_prompt="sys",
        )


# ---------------------------------------------------------------------------
# test_claude_code_handles_cli_missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_code_handles_cli_missing(monkeypatch):
    """_extract_claude_code raises RuntimeError when the CLI is missing/not logged in."""
    fake_sdk = types.ModuleType("claude_agent_sdk")

    class FakeClaudeAgentOptions:
        def __init__(self, model: str = "", system_prompt: str = "") -> None:
            pass

    async def failing_query(prompt, *, options=None):
        raise OSError("claude: command not found")
        # make it an async generator
        yield  # pragma: no cover

    fake_sdk.query = failing_query
    fake_sdk.ClaudeAgentOptions = FakeClaudeAgentOptions
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    from pgkg.ml import _extract_claude_code

    with pytest.raises(RuntimeError, match="claude.*CLI.*installed and logged in"):
        await _extract_claude_code(
            "some text",
            extractor_model="claude-haiku-4-5-20251001",
            max_propositions=20,
            system_prompt="sys",
        )


# ---------------------------------------------------------------------------
# test_extract_propositions_dispatches_to_claude_code
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_propositions_dispatches_to_claude_code(monkeypatch):
    """extract_propositions_async dispatches to _extract_claude_code when provider=claude_code."""
    import pgkg.ml as ml_module

    # A real Settings, not a MagicMock: the model handed to the provider is now
    # resolved by Settings itself, and a mock would have answered with a mock
    # and pinned nothing.
    from pgkg.config import Settings

    settings = Settings(
        llm_provider="claude_code",
        extractor_model="claude-haiku-4-5-20251001",
        _env_file=None,
    )
    monkeypatch.setattr(ml_module, "get_settings", lambda: settings)
    monkeypatch.delenv("PGKG_OFFLINE_EXTRACT", raising=False)

    from pgkg.ml import Proposition

    stub_props = [
        Proposition(
            text="Alice is a scientist.",
            subject="Alice",
            predicate="is",
            object="scientist",
            object_is_literal=False,
        )
    ]
    call_args: list = []

    async def fake_extract_claude_code(chunk_text, *, extractor_model, max_propositions, system_prompt):
        call_args.append((chunk_text, extractor_model, max_propositions))
        return stub_props

    monkeypatch.setattr(ml_module, "_extract_claude_code", fake_extract_claude_code)

    from pgkg.ml import extract_propositions_async

    result = await extract_propositions_async("Alice is a scientist.")

    assert len(call_args) == 1
    assert call_args[0][0] == "Alice is a scientist."
    assert call_args[0][1] == "claude-haiku-4-5-20251001"
    assert result[0].subject == "Alice"


# ---------------------------------------------------------------------------
# test_call_llm_dispatches_to_claude_code
# ---------------------------------------------------------------------------

def test_call_llm_dispatches_to_claude_code(monkeypatch):
    """bench.common._call_llm dispatches to claude_code path and returns text."""
    fake_sdk = _make_fake_sdk("hello from claude")
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    import bench.common as bench_common

    # Provide a settings mock so _call_llm doesn't error on other providers
    fake_settings = MagicMock()
    monkeypatch.setattr(bench_common, "get_settings", lambda: fake_settings)

    result = bench_common._call_llm("test prompt", model="claude-haiku-4-5-20251001", provider="claude_code")
    assert result == "hello from claude"


# ---------------------------------------------------------------------------
# Which model the provider actually asks for
# ---------------------------------------------------------------------------
#
# Found by standing the system up and driving it over MCP: selecting the
# provider on its own sent `gpt-4o-mini-2024-07-18` to the `claude` CLI, which
# failed — and the failure was reported as "install the CLI and log in", so the
# real cause was invisible.  `.env.local-claude` sets both model variables and
# therefore masked it; the README's Path B says only to set the provider.


def test_selecting_claude_code_does_not_ask_it_for_an_openai_model(monkeypatch):
    """The provider default has to follow the provider.

    `llm_model` defaults to an OpenAI id because most callers use OpenAI.  A
    caller who names a provider and no model is asking for that provider's
    default, not for the other one's.
    """
    from pgkg.config import Settings

    settings = Settings(llm_provider="claude_code", _env_file=None)

    assert settings.resolved_extractor_model.startswith("claude-"), (
        "claude_code was handed "
        f"{settings.resolved_extractor_model!r}, which is not a Claude model"
    )


def test_an_explicit_model_still_wins_over_the_provider_default(monkeypatch):
    """The fix must not take the choice away from someone who made one."""
    from pgkg.config import Settings

    explicit = Settings(
        llm_provider="claude_code", llm_model="claude-opus-4-20250514", _env_file=None
    )
    assert explicit.resolved_extractor_model == "claude-opus-4-20250514"

    override = Settings(
        llm_provider="claude_code",
        extractor_model="claude-sonnet-4-20250514",
        _env_file=None,
    )
    assert override.resolved_extractor_model == "claude-sonnet-4-20250514"


def test_the_other_providers_keep_their_own_default(monkeypatch):
    from pgkg.config import Settings

    assert Settings(llm_provider="openai", _env_file=None).resolved_extractor_model == (
        "gpt-4o-mini-2024-07-18"
    )


async def test_a_failing_sdk_call_reports_what_actually_went_wrong(monkeypatch):
    """The old handler replaced every failure with advice about logging in.

    A wrong model id, a network error and a genuinely unauthenticated CLI all
    read identically, which is how the model mismatch above stayed hidden.
    """
    from pgkg import ml

    fake_sdk = types.ModuleType("claude_agent_sdk")

    async def _boom(*args, **kwargs):
        raise ValueError("model 'gpt-4o-mini-2024-07-18' not found")
        yield  # pragma: no cover — makes this an async generator

    fake_sdk.query = _boom
    fake_sdk.ClaudeAgentOptions = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    with pytest.raises(RuntimeError) as caught:
        await ml._extract_claude_code("some text", system_prompt="x")

    message = str(caught.value)
    assert "gpt-4o-mini-2024-07-18" in message, (
        f"the underlying cause is not in the message: {message}"
    )
    assert isinstance(caught.value.__cause__, ValueError)
