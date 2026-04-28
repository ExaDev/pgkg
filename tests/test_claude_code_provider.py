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

    fake_settings = MagicMock()
    fake_settings.llm_provider = "claude_code"
    fake_settings.extractor_model = "claude-haiku-4-5-20251001"
    fake_settings.llm_model = "claude-haiku-4-5-20251001"
    monkeypatch.setattr(ml_module, "get_settings", lambda: fake_settings)
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
