# 02 API Data Ingestion

**Concept**  
Many real-world datasets come from APIs (web endpoints). Use `requests` to fetch JSON data, handling rate limits and errors. For this module, requests is optional.

### Example
```python
try:
    import requests
    # Instead of real remote URL, use a local mock endpoint if possible
    response = requests.get("https://jsonplaceholder.typicode.com/todos/1")
    if response.status_code == 200:
        data = response.json()
        print("Title:", data["title"])
    else:
        print("API error:", response.status_code)
except ImportError:
    print("requests not installed. Run: pip install requests")
```

### Exercise
"""
Fetch the first 5 todos from jsonplaceholder.typicode.com and print their titles. (Requires requests.)
"""
```python
# Your code here
```

### Quiz
**Question:** What data format is most commonly returned by modern APIs?
- A) XML
- B) CSV
- C) JSON
- D) YAML
**Answer:** C