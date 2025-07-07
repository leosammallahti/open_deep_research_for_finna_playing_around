# Open Deep Research - Streamlit UI

## 🚀 Quick Start

### 1. Install Dependencies

First, make sure you have all dependencies installed:

```bash
# Using pip
pip install streamlit

# Or using uv (recommended)
uv pip install streamlit
```

### 2. Set Up API Keys

Create a `.env` file in the root directory with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
# Add other API keys as needed for different search providers
```

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📋 Features

- **🔍 Multiple Search Providers**: Choose from Google, DuckDuckGo, ArXiv, PubMed, Perplexity, and more
- **📄 Adjustable Report Length**: Short (100-150 words), Moderate (150-200 words), or Long (200-300 words) per section
- **🤖 Model Selection**: Use different AI models for planning and research
- **📥 Download Reports**: Export your research as Markdown files
- **🎨 Beautiful UI**: Clean, modern interface with progress tracking

## 🔧 Configuration Options

### Search Providers
- **Google Search**: General web search
- **DuckDuckGo**: Privacy-focused web search
- **ArXiv**: Academic papers in physics, mathematics, computer science
- **PubMed**: Medical and life science research
- **Perplexity**: AI-powered search
- **Tavily**: Research-focused search
- **Exa**: Semantic search engine
- **Linkup**: Real-time web search
- **Azure AI Search**: Enterprise search (requires Azure setup)

### AI Models
- **Claude 3.5 Sonnet**: Best for complex research tasks
- **Claude 3.5 Haiku**: Faster, good for simpler topics
- **GPT-4**: OpenAI's most capable model
- **GPT-3.5 Turbo**: Faster and more cost-effective
- **Groq Llama 3.1**: Open-source alternative

## 💡 Tips for Better Results

1. **Be Specific**: Provide clear, detailed research topics
2. **Choose the Right Search Provider**: 
   - Use ArXiv for academic/scientific topics
   - Use PubMed for medical research
   - Use general search for broader topics
3. **Adjust Report Length**: Start with "moderate" and adjust based on your needs
4. **Enable Clarifications**: Turn on "Allow AI to ask clarifying questions" for complex topics

## 🐛 Troubleshooting

- **"Module not found" error**: Make sure all dependencies are installed
- **API errors**: Check that your API keys are correctly set in the `.env` file
- **Search provider errors**: Some providers require additional API keys (e.g., Google requires setup)
- **Slow performance**: Try using faster models like Claude 3.5 Haiku or GPT-3.5 Turbo

## 📝 Example Topics

- "The impact of artificial intelligence on healthcare diagnostics"
- "Climate change mitigation strategies in urban environments"
- "Recent advances in quantum computing applications"
- "The role of microbiome in human health"
- "Sustainable energy solutions for developing countries" 