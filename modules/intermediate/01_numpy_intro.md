# NumPy Introduction

**Concept**  
NumPy provides fast array operations and is the core of numerical computing in Python. Arrays (`ndarray`) allow vectorized math and broadcasting, enabling efficient computation over large datasets.

### Example
```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
print("Sum:", a + b)
print("Mean of a:", np.mean(a))
print("Elementwise multiply:", a * b)
```

### Exercise
"""
Create a 2D NumPy array and compute the mean of each column.
"""
```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
# Your code here:
```

### Quiz
**Q1:** What is the main benefit of using NumPy arrays over Python lists for numeric data?
- A) Simpler syntax
- B) Faster, vectorized computation
- C) Better for text
- D) Unlimited size
**A:** B

**Q2:** Which function creates an array of zeros in NumPy?
**A:** zeros