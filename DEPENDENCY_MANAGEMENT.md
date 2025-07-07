# 📦 Dependency Management Guide

## 🎯 Overview

Open Deep Research now uses **smart dependency management** to solve the recurring error issues. Instead of requiring all search providers, you can now install only the ones you need!

## 🔧 Installation Options

### **Option 1: Basic Installation (Recommended)**
```bash
pip install open-deep-research
```
This installs:
- ✅ Core functionality (LangGraph, LangChain, etc.)
- ✅ DuckDuckGo search (always available)
- ✅ Basic search capabilities
- ✅ Streamlit UI

### **Option 2: Install Specific Search Providers**
```bash
# Install individual search providers
pip install open-deep-research[tavily]     # Tavily search
pip install open-deep-research[exa]        # Exa semantic search
pip install open-deep-research[arxiv]      # Academic papers
pip install open-deep-research[pubmed]     # Medical research
pip install open-deep-research[google]     # Google search
pip install open-deep-research[azure]      # Azure AI Search
pip install open-deep-research[linkup]     # Linkup real-time search

# Install multiple providers
pip install open-deep-research[tavily,exa,arxiv]
```

### **Option 3: Install All Search Providers**
```bash
pip install open-deep-research[all-search]
```

## 🔍 Search Provider Status

The app now automatically detects which search providers are available and **only shows the ones you can use**.

### **Check Available Providers**
```python
from open_deep_research.dependency_manager import get_status_report
print(get_status_report())
```

### **In Streamlit UI**
- ✅ Available providers are shown in green
- ⚠️ Missing providers show a warning with install instructions
- 📋 Click "View All Provider Status" to see what's available

## 🛠️ How It Works

### **Smart Import System**
- **Before**: All packages were required → Import errors if missing
- **After**: Packages are optional → Graceful fallback with helpful messages

### **Example Error Messages**
**Old (confusing):**
```
ModuleNotFoundError: No module named 'exa_py'
```

**New (helpful):**
```
❌ Exa is not available. To use this search provider, install it with:
  pip install exa-py>=1.8.8
Description: Semantic search engine
```

## 🚀 Quick Start Examples

### **Minimal Setup (Research with DuckDuckGo)**
```bash
pip install open-deep-research
streamlit run streamlit_app.py
```

### **Academic Research Setup**
```bash
pip install open-deep-research[arxiv,pubmed]
streamlit run streamlit_app.py
```

### **Full-Featured Setup**
```bash
pip install open-deep-research[all-search]
streamlit run streamlit_app.py
```

## 📋 Search Provider Details

| Provider | Package | Description | API Key Required |
|----------|---------|-------------|------------------|
| **DuckDuckGo** | `duckduckgo-search` | Privacy-focused web search | ❌ No |
| **Tavily** | `tavily-python` | Research-focused API | ✅ Yes |
| **ArXiv** | `arxiv` | Academic papers | ❌ No |
| **PubMed** | `xmltodict` | Medical literature | ❌ No |
| **Exa** | `exa-py` | Semantic search engine | ✅ Yes |
| **Google** | `googlesearch-python` | Google web search | ❌ No |
| **Perplexity** | Built-in | AI-powered search | ✅ Yes |
| **Linkup** | `linkup-sdk` | Real-time web search | ✅ Yes |
| **Azure AI** | `azure-search-documents` | Enterprise search | ✅ Yes |

## 🐛 Troubleshooting

### **"No search providers available"**
```bash
# Install at least one search provider
pip install open-deep-research[tavily]
```

### **"ImportError: xyz is not available"**
```bash
# Install the specific provider mentioned in the error
pip install open-deep-research[provider-name]
```

### **"All search providers keep failing"**
1. Check your internet connection
2. Verify API keys are set correctly in `.env`
3. Try DuckDuckGo (no API key required)
4. Check the provider status in the UI

## 📊 Migration from Old System

### **If you had the old version:**
```bash
# Uninstall old version
pip uninstall open-deep-research

# Install new version with only providers you need
pip install open-deep-research[tavily,exa]
```

### **If you want everything like before:**
```bash
pip install open-deep-research[all-search]
```

## 🎯 Best Practices

1. **Start minimal**: Install only `open-deep-research` first
2. **Add providers as needed**: Install specific providers when you need them
3. **Use the UI**: Let the app tell you what's available
4. **Check the status**: Use the status report to debug issues

## 🔗 Quick Commands

```bash
# Minimal installation
pip install open-deep-research

# Academic research
pip install open-deep-research[arxiv,pubmed]

# General web research
pip install open-deep-research[tavily,exa,google]

# Everything
pip install open-deep-research[all-search]
```

---

**This new system eliminates the dependency management issues that were causing repeated errors!** 🎉 