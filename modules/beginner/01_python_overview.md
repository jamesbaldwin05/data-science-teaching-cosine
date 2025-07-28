# Python Refresher for Data Science Practitioners

## Who is This Module For?

This advanced refresher is designed for experienced Python programmers who are comfortable with core syntax and programming concepts but are new to data science. If you know your way around functions, classes, and built-in types, this module will help you bridge the gap to data-science-specific Python usage, highlighting language features and ecosystem tools you'll encounter in real-world pipelines.

---

## Prerequisite Python Skills: Self-Assessment Checklist

Before proceeding, ensure you can confidently answer "yes" to the following:

- Can you write and use functions, including with default and keyword arguments?
- Are you comfortable with for/while loops and list/dict/set comprehensions?
- Do you understand Python objects, classes, and basic OOP (inheritance, methods)?
- Can you read and write files, and handle exceptions with try/except?
- Are you familiar with modules, packages, and virtual environments?
- Have you used pip to install Python packages?
- Can you use built-in data structures (lists, dicts, sets, tuples) effectively?
- Have you written or understood basic unit tests?

If you answered "no" to any, consider reviewing Python fundamentals first.

---

# Essential Python Features & Practices for Data Science

A successful data scientist in Python needs fluency with the following language features. Each is presented below with a practical, data-science-flavored example.

---

## 1. Lists & List Comprehensions

**What/Why:** Lists hold collections of items (e.g., rows, feature values). List comprehensions enable concise, readable transformations—ubiquitous in data wrangling.

**Example:**

```python
temperatures_c = [12, 18, 22, 15]
temperatures_f = [c * 9/5 + 32 for c in temperatures_c]
# [53.6, 64.4, 71.6, 59.0]
```

**Depth:** Refresher; expect to write/read nested comprehensions.

---

## 2. Dictionaries & Nested Dicts

**What/Why:** Dicts map keys to values—ideal for label-value pairs, config settings, and JSON-like data.

**Example:**

```python
record = {"name": "Alice", "scores": {"math": 90, "bio": 87}}
mean_score = sum(record["scores"].values()) / len(record["scores"])
```

**Depth:** Refresher; nested dict manipulations common.

---

## 3. Tuples & Unpacking

**What/Why:** Tuples are immutable sequences, used for fixed-size data (e.g., (x, y)), safe function returns, and unpacking.

**Example:**

```python
def min_max(values):
    return min(values), max(values)

lo, hi = min_max([2, 8, 3])
# lo=2, hi=8
```

**Depth:** Refresher; comfortable with tuple unpacking in assignments and loops.

---

## 4. Sets

**What/Why:** Sets store unique items; great for deduplication, membership testing, and set algebra (union/intersection).

**Example:**

```python
labels = ["cat", "dog", "cat", "mouse"]
unique_labels = set(labels)  # {'cat', 'dog', 'mouse'}
```

**Depth:** Refresher.

---

## 5. Slicing & Indexing

**What/Why:** Slicing extracts sublists/sublists—vital for data selection and windowing.

**Example:**

```python
data = [0, 1, 2, 3, 4, 5]
window = data[2:5]  # [2, 3, 4]
reversed_data = data[::-1]
```

**Depth:** Refresher; must be comfortable with multi-level slicing.

---

## 6. Comprehensions vs. Generator Expressions

**What/Why:** Comprehensions (list/dict/set) eagerly build collections; generator expressions produce items lazily—important for efficiency with large data.

**Example:**

```python
squares = (x**2 for x in range(10))  # generator
total = sum(squares)
```

**Depth:** Refresher; able to read/write both patterns.

---

## 7. Functions (including Lambdas)

**What/Why:** Functions organize code; lambdas create small, anonymous functions—handy for sorting, filtering, or passing to library methods.

**Example:**

```python
data = ["apple", "pear", "banana"]
data.sort(key=lambda word: len(word))
# ['pear', 'apple', 'banana']
```

**Depth:** Refresher; use of first-class functions, optional/keyword arguments.

---

## 8. itertools & functools

**What/Why:** These standard libraries provide advanced iteration and functional utilities (e.g., grouping, mapping, accumulating).

**Example:**

```python
from itertools import groupby
data = ["a", "aa", "b", "bb", "b"]
groups = {k: list(g) for k, g in groupby(sorted(data), key=lambda x: x[0])}
# {'a': ['a', 'aa'], 'b': ['b', 'bb', 'b']}
```

**Depth:** Advanced; know the most common tools (chain, groupby, accumulate).

---

## 9. Classes & DataClasses

**What/Why:** Classes encapsulate data/behavior; dataclasses (Python 3.7+) make lightweight record types—popular for clear, type-safe data containers.

**Example:**

```python
from dataclasses import dataclass

@dataclass
class Measurement:
    id: int
    value: float

m = Measurement(1, 3.2)
```

**Depth:** Refresher for classes, able to use dataclasses.

---

## 10. Context Managers

**What/Why:** Use `with` to manage resources (files, DB connections) safely—ensures cleanup even on error.

**Example:**

```python
with open('results.txt', 'w') as f:
    f.write("Experiment complete.")
```

**Depth:** Refresher; able to write custom context managers if needed.

---

## 11. Error Handling

**What/Why:** Proper use of `try`/`except` ensures robust pipelines, especially with unpredictable data.

**Example:**

```python
try:
    val = float("not a number")
except ValueError as e:
    print(f"Conversion failed: {e}")
```

**Depth:** Refresher; comfortable with handling and raising exceptions.

---

## 12. Type Hints & the typing Module

**What/Why:** Adding type hints improves code clarity, enables better IDE support, and supports static analysis.

**Example:**

```python
from typing import List

def mean(values: List[float]) -> float:
    return sum(values) / len(values)
```

**Depth:** Refresher; familiarity with common types (List, Dict, Optional, Union).

---

## 13. Virtual Environments & Dependency Management

**What/Why:** Isolating environments prevents package conflicts and ensures reproducibility.

**Example:**

```sh
python -m venv .env
source .env/bin/activate
pip install numpy pandas
```

**Depth:** Refresher; expected to use venv, pip, and requirements.txt.

---

## 14. Unit Testing (pytest)

**What/Why:** Tests catch regressions and document expectations. `pytest` is the de facto testing library.

**Example:**

```python
def add(x, y):
    return x + y

def test_add():
    assert add(2, 3) == 5
```

**Depth:** Refresher; able to write and run basic pytest functions.

---

## 15. Logging

**What/Why:** Logging (vs. print) enables scalable monitoring, diagnostics, and debugging.

**Example:**

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Data pipeline started")
```

**Depth:** Refresher; able to configure and use logging basics.

---

## 16. Performance Tips (Vectorization, Generators, Memory Views)

**What/Why:** Efficient code is crucial with large data. Prefer vectorized operations (with numpy/pandas), generators for streaming, and memory views for large binary data.

**Example: Vectorization**

```python
import numpy as np
a = np.arange(1000000)
b = a * 2  # vectorized, fast
```

**Depth:** Advanced; know when and how to avoid explicit loops.

---

## 17. Debugging

**What/Why:** Debuggers and assert statements are invaluable for inspecting data flows and catching subtle bugs.

**Example:**

```python
def process(data):
    assert isinstance(data, list), "data must be a list"
    # or use built-in breakpoint()
    # breakpoint()
```

**Depth:** Refresher; comfortable with Python debuggers (pdb, IDE tools).

---

# Python Ecosystem for Data Science

Mastering Python for data science also means knowing the landscape of essential libraries. Each will be covered in-depth later, but here are the must-knows:

---

## Numpy

Numerical arrays, vectorized operations, linear algebra.

```python
import numpy as np
arr = np.array([1, 2, 3])
print(arr.mean())
```

---

## Pandas

Data frames, tabular data wrangling, CSV/Excel/SQL integration.

```python
import pandas as pd
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
print(df.describe())
```

---

## Matplotlib & Seaborn

Visualization libraries for everything from quick plots to complex charts.

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

---

## Scikit-learn

Machine learning algorithms, preprocessing, and model evaluation.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit([[0], [1]], [0, 1])
```

---

## Scipy

Scientific computing: stats, optimization, signal/image processing.

```python
from scipy import stats
print(stats.norm.cdf(0))
```

---

## Statsmodels

Advanced statistical modeling (regression, time series, hypothesis tests).

```python
import statsmodels.api as sm
model = sm.OLS([2, 4, 6], sm.add_constant([1, 2, 3])).fit()
print(model.summary())
```

---

## Jupyter Notebooks

Interactive, literate programming for data exploration and reporting.

```python
# In a notebook cell:
print("Hello, Data Science!")
```

---

# Key Takeaways & Next Steps

- Data science with Python builds on solid core language skills—review as needed!
- Mastering lists, dicts, comprehensions, generators, and classes is essential for readable, high-performance code.
- Key ecosystem libraries like numpy, pandas, matplotlib, and scikit-learn are foundational—learn their idioms and APIs.
- Robust workflow includes using virtual environments, testing, logging, and performance-aware coding.
- Next: Dive into each core library module by module, starting with [Numpy](../intermediate/01_numpy_intro.md).