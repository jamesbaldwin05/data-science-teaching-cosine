# 01 Python Overview for Data Science

**Concept**  
Python is the dominant language for data science due to its readability and rich ecosystem. Key skills include working with lists, dictionaries, functions, and using libraries (like `pandas`, `numpy`). For more, see [Official Python Tutorial](https://docs.python.org/3/tutorial/) and [Real Python](https://realpython.com/).

### Example
```python
# List comprehension and dictionary use
squares = [x**2 for x in range(5)]
print("Squares:", squares)
info = {"name": "Ada", "age": 30}
print("Name:", info["name"])
```

### Exercise
"""
Write a function that takes a list of numbers and returns their mean (average).
"""
```python
def mean(nums):
    # Your code here
    pass

print(mean([1, 2, 3, 4, 5]))
```

### Quiz
**Question:** What is the recommended way to install external Python libraries?
- A) python install somepackage
- B) pip install somepackage
- C) import install somepackage
- D) setup.py install somepackage
**Answer:** B