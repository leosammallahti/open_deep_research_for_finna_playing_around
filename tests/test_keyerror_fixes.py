from open_deep_research.utils import deduplicate_and_format_sources, safe_get


class TestSafeGet:
    def test_dict_access(self):
        assert safe_get({"key": "value"}, "key") == "value"
        assert safe_get({"key": "value"}, "missing", "default") == "default"

    def test_object_access(self):
        class Obj:
            attr = "value"

        obj = Obj()
        assert safe_get(obj, "attr") == "value"
        assert safe_get(obj, "missing", "default") == "default"

    def test_none_input(self):
        assert safe_get(None, "key", "default") == "default"


class TestDeduplicateAndFormatSources:
    def test_empty_input(self):
        assert "No search results found" in deduplicate_and_format_sources([])
        assert "No search results found" in deduplicate_and_format_sources(None)

    def test_malformed_data(self):
        result = deduplicate_and_format_sources([{"no_results": True}])
        assert isinstance(result, str)

    def test_missing_fields(self):
        result = deduplicate_and_format_sources(
            [{"results": [{"url": "http://test.com"}]}]
        )
        assert "Untitled" in result
        assert "http://test.com" in result
