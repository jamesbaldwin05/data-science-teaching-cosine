# Python in Data Science Overview

## Who is This Course For?

This course is designed for intermediate to experienced Python programmers who are comfortable with core syntax and programming concepts but are new to data science. If you know your way around functions, classes, and built-in types, this module will help you bridge the gap to data-science-specific Python usage, highlighting language features and ecosystem tools (including libraries) you'll encounter in real-world pipelines.

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

### Where to Brush Up

Need to refresh your Python basics? Here are some excellent resources:

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Harvard CS50’s Introduction to Programming with Python](https://cs50.harvard.edu/python/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python – Python Basics](https://realpython.com/learn/python-first-steps/)
- [Python for Everybody (Coursera)](https://www.coursera.org/specializations/python)

---

# Essential Python Features & Practices for Data Science

A successful data scientist in Python needs fluency with the following language features.

---

## 1. Lists & List Comprehensions

**What/Why:** Lists are used in data science to store collections of data, such as rows in a dataset or individual feature values. List comprehensions provide a clear and compact way to create or transform these lists, making data manipulation tasks faster and more readable—an essential part of data wrangling.

**Python lists vs. C-style arrays:** Unlike C arrays, Python lists can store elements of any type and can grow or shrink dynamically. For large numerical arrays, consider using the `array` module (for numbers) or, more commonly in data science, NumPy arrays for efficiency and extra functionality.

**Example:**

```python
temperatures_c = [12, 18, 22, 15]
temperatures_f = [c * 9/5 + 32 for c in temperatures_c]
print(temperatures_f)
```

**You should be able to:**  
- Create, access, and modify lists  
- Write and read list comprehensions (including nested comprehensions)  
- Recognize when to use lists versus numpy arrays for large, homogeneous data

---

## 2. Dictionaries & Nested Dicts

**What/Why:** Dicts map keys to values—ideal for label-value pairs, config settings, and JSON-like data. Nested dicts are common when representing structured or hierarchical information.

**Example:**

```python
record = {"name": "Alice", "scores": {"math": 90, "bio": 87}}
mean_score = sum(record["scores"].values()) / len(record["scores"])
print(mean_score)
```

**Nested dictionary manipulation:**

```python
record = {"name": "Alice", "scores": {"math": 90, "bio": 87}}
# Add a new subject score
record["scores"]["chem"] = 85

# Update a value
record["scores"]["math"] += 5

# Iterate over nested dictionary
for subject, score in record["scores"].items():
    print(f"{subject}: {score}")

print(record["scores"])
```
*This pattern is common when processing JSON data or configuring experiments.*

**You should be able to:**  
- Create, access, update, and iterate over (nested) dictionaries  
- Handle missing keys gracefully  
- Structure and manipulate JSON-like data

---

## 3. Tuples & Unpacking

**What/Why:** Tuples are immutable sequences — once created, their contents cannot be changed. They’re ideal for fixed-size, heterogeneous data (e.g. coordinate pairs, function returns). Tuples support multiple assignment and “starred unpacking” for flexible splitting.

**Tuple immutability:**  
Once a tuple is created, you cannot change its contents:
```python
# no-run
point = (2, 3)
# point[0] = 5  # Raises TypeError!
```

**Multiple assignment and starred unpacking:**

```python
# no-run
a, b = (1, 2)  # multiple assignment
# a=1, b=2

first, *rest = [10, 20, 30, 40]
# first=10, rest=[20, 30, 40]

x, y, *others = (5, 6, 7, 8)
# x=5, y=6, others=[7, 8]
```

**Example:**

```python
def min_max(values):
    return min(values), max(values)

lo, hi = min_max([2, 8, 3])
```

**You should be able to:**  
- Use tuples for multiple assignment, fixed-size records, and safe return values  
- Unpack tuples (including using `*` for flexible assignment)  
- Recognize tuple immutability and its implications

---

## 4. Sets

**What/Why:** Sets store unique items; great for deduplication, membership testing, and set algebra (union/intersection). Used often for label sets, removing duplicates, or fast lookups.

**Example:**

```python
labels = ["cat", "dog", "cat", "mouse"]
unique_labels = set(labels)
print(unique_labels)
```

**You should be able to:**  
- Create sets from lists or other iterables  
- Perform set operations (union, intersection, difference)  
- Use sets for fast membership tests and deduplication

---

## 5. Slicing & Indexing

**What/Why:** Slicing extracts sublists or substrings—vital for data selection and windowing.

**Multi-level slicing:**  
You can slice lists of lists (matrices) or strings:

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row = matrix[1]      # [4, 5, 6]
submatrix = [row[1:] for row in matrix[:2]]  # [[2, 3], [5, 6]]

text = "data science"
print(text[5:12])
```

**Example:**

```python
data = [0, 1, 2, 3, 4, 5]
window = data[2:5]
reversed_data = data[::-1]
```

**You should be able to:**  
- Use slicing for lists, strings, and (with libraries) numpy arrays  
- Read and write multi-level slices  
- Understand step and negative indexing

---

## 6. Comprehensions vs. Generator Expressions

**What/Why:**  
Comprehensions (list/dict/set) eagerly build collections in memory; generator expressions produce items lazily—crucial for efficiency when dealing with large or streaming data sets.

**Memory usage & lazy evaluation:**  
- List comprehensions store the entire result in memory—fast for small/medium data, potentially inefficient for huge data.
- Generator expressions yield one result at a time using lazy evaluation—consume less memory, suitable for pipelines or iterating over large data sets.

**When to use:**  
- Use comprehensions for transformations where you need the entire result at once.
- Use generators when you process data one item at a time, or the result would be too large to fit in memory.

**Example:**

```python
squares = (x**2 for x in range(10))  # generator
total = sum(squares)
print(total)
```

**You should be able to:**  
- Choose between comprehensions and generators based on data size and memory constraints  
- Write both styles for data wrangling tasks  
- Recognize generator exhaustion

---

## 7. Functions (including Lambdas)

**What/Why:** Functions organize code and logic; lambdas create small, anonymous functions—handy for sorting, filtering, or passing as arguments.

**Positional vs. keyword arguments, defaults, and lambdas:**  
- Positional arguments are matched by position; keyword arguments are matched by name.
- You can specify default values for arguments, which users can override.
- Lambdas are single-expression, anonymous functions (useful as arguments).

**Example:**

```python
def greet(name, msg="Hello"):
    return f"{msg}, {name}!"

print(greet("Data Scientist"))  # uses default msg
print(greet("Data Scientist", msg="Welcome"))

data = ["apple", "pear", "banana"]
data.sort(key=lambda word: len(word))
# ['pear', 'apple', 'banana']
```

**You should be able to:**  
- Define and call functions with positional, keyword, and default arguments  
- Use lambdas in appropriate contexts  
- Pass functions as arguments (first-class functions)

---

## 8. itertools & functools

**What/Why:** These standard libraries provide advanced iteration and functional utilities (e.g., grouping, mapping, accumulating). Must-know tools include:

- `itertools.chain`: concatenate iterables  
- `itertools.product`: cartesian product  
- `itertools.combinations`: all possible pairs/groups  
- `itertools.groupby`: group consecutive items  
- `functools.reduce`: accumulate a result  
- `functools.partial`: pre-fill function args

**Examples:**

```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
# product = 24
print(product)
```

```python
from itertools import groupby, chain
data = ["a", "aa", "b", "bb", "b"]
groups = {k: list(g) for k, g in groupby(sorted(data), key=lambda x: x[0])}
# {'a': ['a', 'aa'], 'b': ['b', 'bb', 'b']}
```

**You should be able to:**  
- Use chain, product, combinations, groupby for iteration tasks  
- Use reduce and partial for custom accumulation/function manipulation  
- Recognize when built-in alternatives are preferable (e.g., sum/min/max)

---

## 9. Classes & DataClasses

**What/Why:**  
- *Classes* encapsulate data and methods, supporting inheritance and customization.
- *DataClasses* (Python 3.7+) are specialized classes for simple data containers, generating boilerplate (e.g., `__init__`, `__repr__`) automatically.

Below are two equivalent ways to define a simple data container: a classic class and a dataclass.

### Classic class example

```python
class Measurement:
    def __init__(self, id, value):
        self.id = id
        self.value = value

m = Measurement(1, 3.2)
print(m)
```

### Dataclass equivalent

```python
from dataclasses import dataclass

@dataclass
class Measurement:
    id: int
    value: float

m = Measurement(1, 3.2)
print(m)
```

**You should be able to:**  
- Write and use custom classes for encapsulating data/behavior  
- Use dataclasses for simple record types  
- Decide when to use each based on project needs

---

## 10. Context Managers

**What/Why:** Use `with` to manage resources (files, DB connections) safely—ensures cleanup even on error.

**How it works:**  
Context managers implement the `__enter__` and `__exit__` methods, which setup and teardown resources automatically. You can write them as classes or using `contextlib.contextmanager`.

**Example:**

```python
# no-run
with open('results.txt', 'w') as f:
    f.write("Experiment complete.")
```

**Custom context manager with contextlib:**

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    print("Resource acquired")
    try:
        yield
    finally:
        print("Resource released")

with managed_resource():
    print("Do work")
```

**You should be able to:**  
- Use `with` for file/database/network resource management  
- Recognize and implement custom context managers (class or decorator style)  
- Understand the role of `__enter__` and `__exit__`

---

## 11. Error Handling

**What/Why:** Proper use of `try`/`except` ensures robust pipelines, especially with unpredictable data.

**Catching specific exceptions, raising new ones, and using all blocks:**

```python
try:
    val = float("not a number")
except ValueError as e:
    print(f"Conversion failed: {e}")
    raise RuntimeError("Parsing error") from e
else:
    print("Conversion succeeded!")
finally:
    print("Cleanup actions (if any)")
```

- `except` can catch specific exceptions for targeted error handling.
- `else` runs if no exception was raised.
- `finally` always runs, useful for cleanup.

**You should be able to:**  
- Catch specific exceptions and raise new ones appropriately  
- Use try/except/else/finally for error-prone code  
- Write robust, user-friendly error messages

---

## 12. Type Hints & the typing Module

**What/Why:**  
- Type hints clarify intent, improve code readability, and enable static analysis tools (e.g., mypy, IDEs).
- The `typing` module offers generics (e.g., `List[T]`, `Dict[K, V]`), `Optional`, `Union`, and more.
- Static analysis can catch bugs before runtime and aid code navigation in large projects.

**Examples:**

```python
from typing import List, Optional, Union, TypedDict

def mean(values: List[float]) -> float:
    return sum(values) / len(values)

def parse_score(val: Union[str, float]) -> Optional[float]:
    try:
        return float(val)
    except ValueError:
        return None

class PersonDict(TypedDict):
    name: str
    age: int

print(mean([1.0, 2.0, 3.0]))
```

**You should be able to:**  
- Add type hints to functions, classes, and variables  
- Use common generics (`List`, `Dict`, `Optional`, `Union`) and TypedDict for structured data  
- Run static type checks with mypy or similar tools

---

## 13. Virtual Environments & Dependency Management

**What/Why:** Isolating environments prevents package conflicts and ensures reproducibility.

- Use `python -m venv` or `conda` to create isolated environments.
- Use `pip install -r requirements.txt` to install dependencies from a file.
- Use `pip install --upgrade package` to update.
- For modern projects, `pyproject.toml` is also gaining traction for dependency management.

**Example:**

```sh
python -m venv .env
source .env/bin/activate
pip install numpy pandas
pip install -r requirements.txt
pip install --upgrade matplotlib
```
`requirements.txt` and `pyproject.toml` help standardize and share dependencies.

**You should be able to:**  
- Create/activate/deactivate virtual environments  
- Install, upgrade, and freeze dependencies using pip  
- Maintain requirements.txt or pyproject.toml

---

## 14. Unit Testing (pytest)

**What/Why:** Tests catch regressions and document expectations. `pytest` is the de facto testing library.

**Example (arrange-act-assert):**

```python
# no-run
# test_math.py
def add(x, y):
    return x + y

def test_add():
    # Arrange
    x, y = 2, 3
    # Act
    result = add(x, y)
    # Assert
    assert result == 5
```

- Fixtures in pytest help you set up reusable test data and environments.

**You should be able to:**  
- Write simple unit tests using pytest  
- Understand arrange/act/assert structure  
- Use and recognize pytest fixtures

---

## 15. Logging

**What/Why:** Logging (vs. print) enables scalable monitoring, diagnostics, and debugging. Logging supports different levels (INFO, WARNING, ERROR), structured logs, and integrates with ML pipeline tools and experiment tracking.

**Example:**

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Data pipeline started")
logging.warning("Missing value encountered in column 'age'")
```

- Logging is configurable, can be captured to files, and is essential for traceability in production and research code.

**You should be able to:**  
- Replace print statements with logging  
- Use different log levels appropriately  
- Integrate logging with larger data workflows

---

## 16. Performance Tips (Vectorization, Generators, Memory Views)

**Vectorization:**  
Perform operations on whole collections (arrays) at once using numpy/pandas. This avoids slow Python loops and leverages fast, compiled code underneath.

```python
import numpy as np
a = np.arange(1000000)
b = a * 2  # vectorized, fast
```
*Why crucial:* Vectorization massively speeds up code, especially with large datasets.

**Generators:**  
Generators yield one item at a time, allowing you to process data streams or large files without loading everything into memory.

```python
def gen():
    for i in range(1_000_000):
        yield i**2

for val in gen():
    if val > 100:
        break
```
*Why crucial:* Prevents memory overload and enables pipeline-style processing.

**Memory Views:**  
Memory views allow you to work efficiently with large binary data (e.g., images or raw byte buffers), avoiding unnecessary copying.

```python
buf = bytearray(b"abcdefgh")
view = memoryview(buf)
print(view[2:5].tobytes())  # b'cde'
```
*Why crucial:* Enables fast, zero-copy operations on large datasets.

**You should be able to:**  
- Use vectorized operations for numerical data  
- Write and consume generators for large/streamed data  
- Utilize memoryview for advanced binary data tasks

---

## 17. Debugging

**What/Why:** Debuggers (`pdb`, IDE tools) and assert statements are invaluable for inspecting data flows and catching subtle bugs.

**Example:**

```python
def process(data):
    assert isinstance(data, list), "data must be a list"
    # or use built-in breakpoint()
    # breakpoint()
```

**You should be able to:**  
- Use assert for sanity checks  
- Set breakpoints and step through code  
- Use pdb or IDE debuggers for troubleshooting

---

# Python Ecosystem for Data Science

These libraries are introduced briefly here — dedicated modules later will dive deep.

Mastering Python for data science also means knowing the landscape of essential libraries. Each will be covered in-depth later, but here are the must-knows:

---

## Numpy

Numpy is the foundational package for numerical computing in Python, providing fast, efficient arrays and a wealth of mathematical functions. It’s essential for scientific and data-intensive work, powering the core of nearly every other data science library in the Python ecosystem.

Numpy’s ndarrays enable vectorized, element-wise operations and support broadcasting, making them far more powerful and efficient than native Python lists. Key submodules include `numpy.linalg` for linear algebra, `numpy.random` for random sampling, and `numpy.fft` for signal processing. Common use-cases include numerical simulations, matrix operations, and rapid prototyping of algorithms.

```python
import numpy as np

# Create a 2D array and perform vectorized operations
data = np.arange(6).reshape(2, 3)
mean_by_column = data.mean(axis=0)
centered = data - mean_by_column
print(centered)

# Linear algebra: matrix multiplication and eigendecomposition
product = data @ data.T
eigenvalues, eigenvectors = np.linalg.eig(product)
print("Eigenvalues:", eigenvalues)
```

---

## Pandas

Pandas is the go-to library for working with structured (tabular) data in Python, offering intuitive and powerful tools for data wrangling, cleaning, and analysis. Its DataFrame and Series objects make it easy to handle everything from small datasets to large-scale data manipulations.

Key capabilities include data alignment, grouping (via `groupby`), time series support, merging/joining datasets, and seamless reading/writing to formats like CSV, Excel, and SQL. Typical use-cases involve cleaning messy data, summarizing statistics, reshaping tables (pivot, melt), and preparing features for machine learning.

```python
import pandas as pd

# Create and inspect a DataFrame
df = pd.DataFrame({'group': ['A', 'A', 'B', 'B'], 'value': [10, 15, 10, 25]})
summary = df.groupby('group').agg(['mean', 'sum'])

# Handle missing values and filter data
df.loc[2, 'value'] = None
df['value'] = df['value'].fillna(df['value'].mean())
filtered = df[df['value'] > 12]

print(filtered)
```

---

## Matplotlib & Seaborn

Matplotlib and Seaborn are the cornerstones of data visualization in Python. Matplotlib provides granular control over every aspect of a figure, suitable for both quick plots and publication-quality graphics. Seaborn builds on Matplotlib with a high-level API and attractive default styles, making statistical plots and data exploration faster and more intuitive.

With these libraries, you can create a wide range of visualizations: scatter plots, line charts, barplots, heatmaps, and regression plots. Seaborn’s integration with pandas DataFrames enables rapid EDA (exploratory data analysis), while Matplotlib’s flexibility allows for custom layouts, annotations, and advanced figures.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load a built-in dataset and visualize relationships
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
sns.regplot(data=tips, x="total_bill", y="tip", scatter=False, color="red")

plt.title("Tip vs. Total Bill")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.tight_layout()
plt.show()
```

---

## Scikit-learn

Scikit-learn is the standard library for machine learning in Python, offering accessible APIs for a wide range of models and tools. It empowers you to quickly build, train, and evaluate machine learning pipelines, covering tasks from simple regression to complex classification and clustering.

The library includes modules for preprocessing (scaling, encoding), feature selection, model selection (train/test split, cross-validation), and metrics. Pipelines make it easy to chain transformations and estimators, ensuring reproducible ML workflows. Common use-cases include rapid prototyping, automated model tuning, and benchmarking algorithms on standard datasets.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Load data and build a pipeline: scale → fit model
X, y = load_breast_cancer(return_X_y=True)
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=0))
scores = cross_val_score(pipeline, X, y, cv=5)

print(f"Mean CV accuracy: {scores.mean():.2%}")
```

---

## Scipy

Scientific computing: stats, optimization, signal/image processing. Scipy extends numpy with a vast library of high-level scientific algorithms and utilities for statistics, optimization, integration, interpolation, and more.

Scipy is organized into key submodules such as `scipy.optimize` (optimization and curve fitting), `scipy.stats` (statistical tests and distributions), `scipy.integrate` (numerical integration), `scipy.signal` (signal processing), and `scipy.sparse` (sparse matrices). Typical uses include fitting models to data, running hypothesis tests, integrating functions, or working with large, sparse datasets.

```python
from scipy import stats
print(stats.norm.cdf(0))

# Example: Student's t-test for independent samples
a = [1.1, 2.5, 3.3]
b = [0.9, 2.1, 3.0]
t_stat, p_value = stats.ttest_ind(a, b)
print(f"t={t_stat:.3f}, p={p_value:.3g}")
```

---

## Statsmodels

Advanced statistical modeling (regression, time series, hypothesis tests). Statsmodels bridges the gap between pure machine learning and statistical inference, providing deep tools for classical statistics and model diagnostics.

Statsmodels features a user-friendly formula API (like R's formulas) for specifying models with symbolic syntax (e.g., `"y ~ x1 + x2"`), as well as advanced tools for time-series analysis (ARIMA, SARIMAX, state space models). Its emphasis on statistical inference means you get p-values, confidence intervals, and detailed summaries—ideal for understanding model significance and diagnostics.

```python
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({"y": [2, 4, 6], "x": [1, 2, 3]})
model = smf.ols("y ~ x", data=df).fit()
print(model.summary())
```

---

## Jupyter Notebooks

Jupyter Notebooks are powerful, interactive documents that combine live code, visualizations, and explanatory text. Widely used in data science, teaching, and research, notebooks allow you to interleave executable code (Python or other languages), formatted notes (Markdown), equations, and rich outputs (plots, tables, images) in a single, shareable file. Each notebook consists of "cells"—the two most common types are **code cells** (which you run to produce output) and **Markdown cells** (for formatted text, math, or instructions).

When working in a notebook, you execute code cells individually, and the results—such as printed output, plots, or error messages—appear immediately below the cell. The underlying process that runs your code is called a **kernel**, which maintains the execution state and variables across cells (so earlier results can be reused later). Restarting the kernel clears this state. Notebooks are ideal for exploratory data analysis (EDA), rapid prototyping, teaching, and sharing reproducible reports or data workflows.

# Key Takeaways

- Data science with Python builds on solid core language skills—review as needed!
- Mastering lists, dicts, comprehensions, generators, and classes is essential for readable, high-performance code.
- Key ecosystem libraries like numpy, pandas, matplotlib, and scikit-learn are foundational—learn their idioms and APIs.
- Robust workflow includes using virtual environments, testing, logging, and performance-aware coding.

### Exercise
"""
Generate a list containing the squares of numbers 1-10 using a list comprehension, then print the result.
"""
```python
# Write your solution below
squares = [
    # TODO: your code here
]
print(squares)
```