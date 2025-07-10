import pytest

from open_deep_research.core.format_utils import safe_context


@pytest.mark.asyncio
async def test_safe_context_truncates():
    # Create long string > 24000 chars (~6000 tokens)
    long_str = "x" * 30000
    result = await safe_context(long_str, target_model="gpt-3.5-turbo")

    assert len(result) <= 24000
    # Should be last characters (tail kept)
    assert result == long_str[-24000:]


@pytest.mark.asyncio
async def test_safe_context_no_change():
    short_str = "hello world"
    result = await safe_context(short_str)
    assert result == short_str
