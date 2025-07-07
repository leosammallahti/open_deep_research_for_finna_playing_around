# Configuration Guide for Open Deep Research

## Quick Start

1. Create a `.env` file in the project root directory
2. Add your API keys (see examples below)
3. Run the application

## Required API Keys

You need **at least one** model provider API key to run the application:

### Model Providers

- **Anthropic** (Recommended): `ANTHROPIC_API_KEY=sk-ant-...`
  - Provides Claude models (best overall performance)
  - Get your key at: https://console.anthropic.com/

- **OpenAI**: `OPENAI_API_KEY=sk-...`
  - Provides GPT models (good balance of cost/performance)
  - Get your key at: https://platform.openai.com/api-keys

- **DeepSeek**: `DEEPSEEK_API_KEY=sk-...`
  - Provides DeepSeek models (strong reasoning capabilities)
  - Get your key at: https://platform.deepseek.com/

### Search Providers (Optional but Recommended)

- **Tavily**: `TAVILY_API_KEY=tvly-...`
  - Best for general web search
  - Get your key at: https://tavily.com/

- **DuckDuckGo**: No API key needed (free fallback option)

## Example .env File

```bash
# Create a file named .env (not .env.example) with your keys:

# Model Provider (at least one required)
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
DEEPSEEK_API_KEY=sk-your-key-here

# Search Provider (optional but recommended)
TAVILY_API_KEY=tvly-your-key-here

# Optional Configuration
SEARCH_API=tavily
NUMBER_OF_QUERIES=2
```

## Available Model Combinations

Based on your API keys, different model combinations will be available:

1. **Balanced (Claude 3.5 Sonnet)** - Requires `ANTHROPIC_API_KEY`
2. **Budget (GPT-3.5 Turbo)** - Requires `OPENAI_API_KEY`
3. **Max Reasoning (DeepSeek + Claude)** - Requires both `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY`
4. **All Open-Source (Llama 3)** - Requires `GROQ_API_KEY`

## Troubleshooting

### "No model combinations available" Error
- Make sure you have at least one valid API key in your `.env` file
- Check that the `.env` file is in the project root directory
- Verify your API keys are correct and have available credits

### "API key not found" Errors
- The application is trying to use a model provider you haven't configured
- Solution: Choose a different model combination or add the missing API key

### Testing Your Configuration

Run the test script to verify your setup:

```bash
python test_api_availability.py
```

This will show:
- Which API keys are configured
- Available model combinations
- Which models can be used for each role

## Minimal Setup

For the simplest setup with good results:

1. Get an Anthropic API key from https://console.anthropic.com/
2. Create `.env` file with:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```
3. Run the app - it will use the "Balanced" model combination

## Cost Considerations

- **Most Expensive**: Claude 3.5 Sonnet (best quality)
- **Balanced**: Mix of Claude Sonnet and Haiku
- **Budget**: GPT-3.5 Turbo (lowest cost, good quality)
- **Free Search**: DuckDuckGo (no API key needed) 