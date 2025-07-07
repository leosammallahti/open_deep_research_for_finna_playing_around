# Budget-Friendly Configuration Guide

## ✅ **Issues Fixed**

**Note for future assistance:** The following issues from previous error logs have been resolved:

1. **SearchQuery TypeError Fixed**: The `TypeError: 'SearchQuery' object is not subscriptable` error in `src/open_deep_research/graph.py` line 308 has been resolved. The code now correctly uses `q.search_query` instead of `q["search_query"]`.

2. **API Key Issues Resolved**: Missing API key errors (GROQ_API_KEY, etc.) have been fixed.

3. **Model Name Typo Fixed**: The model name typo `claude-3.5-sonnet-20240620` has been corrected to `claude-3-5-sonnet-20240620` in:
   - `src/open_deep_research/configuration.py`
   - `src/open_deep_research/model_registry.py`  
   - `simple_test.py`

---

## 🆓 **Free Search Providers**
Use these completely free options to avoid search API costs:

```bash
# In your .env file
SEARCH_API=duckduckgo  # Free web search
# OR
SEARCH_API=arxiv       # Free academic papers
# OR  
SEARCH_API=pubmed      # Free medical research
```

## 💰 **Cost-Effective Model Options**

### Option 1: Claude Haiku (Cheapest Anthropic Model)
```bash
PLANNER_PROVIDER=anthropic
PLANNER_MODEL=claude-3-haiku-20240307  # ~10x cheaper than Sonnet
WRITER_PROVIDER=anthropic  
WRITER_MODEL=claude-3-haiku-20240307
```

### Option 2: Groq (Fast + Affordable)
```bash
PLANNER_PROVIDER=groq
PLANNER_MODEL=llama-3.1-70b-versatile
WRITER_PROVIDER=groq
WRITER_MODEL=llama-3.1-70b-versatile
```

### Option 3: DeepSeek (Very Cheap)
```bash
PLANNER_PROVIDER=deepseek
PLANNER_MODEL=deepseek-chat
WRITER_PROVIDER=deepseek
WRITER_MODEL=deepseek-chat
```

## 🐌 **Rate Limiting Solutions**

### 1. Increase Search Delays
Your system already has smart rate limiting. You can tune these in your code:

```python
# In utils.py, these delays are already implemented:
# - Exa: 0.25s between requests (4 req/sec)
# - ArXiv: 3.0s between requests (1 req/3sec)
# - PubMed: 1.0s between requests (adaptive)
# - DuckDuckGo: 2-4s between requests (with backoff)
```

### 2. Reduce Concurrent Requests
```python
# Modify search functions to use fewer concurrent requests
max_concurrent_searches = 2  # Instead of 5
```

### 3. Use Fewer Search Queries
```python
# In your configuration, reduce the number of queries generated
max_search_queries = 3  # Instead of 5-10
```

## 🚀 **Quick Start for Budget Mode**

1. **Set your .env file:**
```bash
SEARCH_API=duckduckgo
PLANNER_PROVIDER=anthropic
PLANNER_MODEL=claude-3-haiku-20240307
WRITER_PROVIDER=anthropic
WRITER_MODEL=claude-3-haiku-20240307
ANTHROPIC_API_KEY=your_key_here
```

2. **Run with budget settings:**
```bash
python streamlit_app.py
```

## 📊 **Cost Comparison**

| Model | Cost per 1M tokens | Speed | Quality |
|-------|-------------------|-------|---------|
| Claude Haiku | $0.25 | Fast | Good |
| Groq Llama 3.1 | $0.59 | Very Fast | Good |
| DeepSeek Chat | $0.14 | Medium | Good |
| Claude Sonnet | $3.00 | Medium | Excellent |

## 🔧 **Advanced Rate Limiting**

If you still hit rate limits, consider these modifications:

### 1. Implement Exponential Backoff
```python
# Already implemented in your utils.py
delay = backoff_factor ** retry_count + random.random()
```

### 2. Use Queue-Based Processing
```python
# Process searches sequentially instead of concurrently
for query in search_queries:
    result = await search_function(query)
    await asyncio.sleep(delay)
```

### 3. Implement Token Bucket Rate Limiting
```python
# Add token bucket algorithm for more sophisticated rate limiting
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
```

## 🎯 **Recommended Budget Setup**

For minimal costs while maintaining functionality:

1. **Search**: DuckDuckGo (free)
2. **Models**: Claude Haiku for both planner and writer
3. **Rate Limiting**: Use existing delays, increase if needed
4. **Queries**: Limit to 3 search queries per section

This should give you a working system for under $1-2 per research session. 