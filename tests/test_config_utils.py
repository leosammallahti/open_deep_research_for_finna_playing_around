from open_deep_research.core.config_utils import get_search_params


def test_get_search_params_filters_unknown_keys():
    config = {
        "max_results": 10,
        "topic": "ai",
        "foo": "bar",  # should be filtered
    }
    params = get_search_params("tavily", config)
    assert "max_results" in params and "topic" in params
    assert "foo" not in params


def test_get_search_params_unknown_api_returns_empty():
    params = get_search_params("unknown_api", {"a": 1})
    assert params == {} 