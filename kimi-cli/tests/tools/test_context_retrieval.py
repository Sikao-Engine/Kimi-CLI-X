"""Tests for ContextRetrieval tool (Phase 3)."""

from __future__ import annotations

import pytest
from kosong.message import Message

from kimi_cli.soul.history_index import HistoryIndex
from kimi_cli.tools.context_retrieval import ContextRetrieval, Params
from kimi_cli.wire.types import TextPart


class TestContextRetrievalTool:
    """Test the ContextRetrieval agent tool."""

    @pytest.fixture
    def tool(self):
        idx = HistoryIndex()
        idx.index_messages([
            Message(role="user", content=[TextPart(text="What did we decide about the API design?")]),
            Message(role="assistant", content=[TextPart(text="We decided to use REST over GraphQL.")]),
            Message(role="user", content=[TextPart(text="How about authentication?")]),
            Message(role="assistant", content=[TextPart(text="We will use OAuth2 with PKCE.")]),
        ])
        return ContextRetrieval(idx)

    @pytest.mark.asyncio
    async def test_search_finds_relevant_turns(self, tool: ContextRetrieval):
        result = await tool(Params(query="API design", k=2))
        assert "REST" in result.output
        assert result.message == "Found 2 turn(s)"

    @pytest.mark.asyncio
    async def test_search_no_results(self, tool: ContextRetrieval):
        result = await tool(Params(query="microservices architecture", k=2))
        assert "No matching past turns found" in result.output

    @pytest.mark.asyncio
    async def test_search_returns_verbatim(self, tool: ContextRetrieval):
        result = await tool(Params(query="authentication", k=3))
        # Either the user query or the assistant answer about OAuth2 should be returned
        assert "authentication" in result.output or "OAuth2" in result.output

    @pytest.mark.asyncio
    async def test_search_shows_compacted_marker(self, tool: ContextRetrieval):
        tool._history_index.mark_compacted()
        result = await tool(Params(query="REST", k=1))
        assert "[compacted]" in result.output

    @pytest.mark.asyncio
    async def test_k_parameter_limits_results(self, tool: ContextRetrieval):
        result = await tool(Params(query="What", k=1))
        # Should only return 1 turn even though multiple match
        assert result.message == "Found 1 turn(s)"
