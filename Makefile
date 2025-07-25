.PHONY: install lint format type test update-lock studio

# Use uv for package management, ensure uv is installed

install:
	uv pip install -e .[dev]

lint:
	ruff check src tests

format:
	ruff format src tests

# Run static type checks
type:
	mypy src tests

# Run the test suite with coverage (HTML + terminal report)
test:
	pytest --cov=src --cov-report=term-missing --cov-report=html

# Update lock file (uv.lock)
update-lock:
	uv pip compile --output-file uv.lock pyproject.toml

# Start LangGraph Studio for development
studio:
	@echo "🚀 Starting LangGraph Studio..."
	@echo "🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
	@echo "📚 API Docs: http://127.0.0.1:2024/docs"
	@echo ""
	langgraph dev --port 2024 