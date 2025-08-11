# Python in Data Science

## Who is This Course For?

This course is designed for intermediate to experienced Python programmers who are comfortable with core syntax and programming concepts but are new to data science. If you know your way around functions, classes, and built-in types, this course will help you bridge the gap to Python usage within data science, highlighting language features and ecosystem tools (including libraries) you'll encounter in real-world pipelines.

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

If you answered "no" to any, consider reviewing Python fundamentals first:
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [Harvard CS50’s Introduction to Programming with Python](https://cs50.harvard.edu/python/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
- [Real Python – Python Basics](https://realpython.com/learn/python-first-steps/)
- [Python for Everybody (Coursera)](https://www.coursera.org/specializations/python)

---

## Table of Contents
1. [Lists & List Comprehensions](#1-lists--list-comprehensions)
2. [Dictionaries & Nested Dicts](#2-dictionaries--nested-dicts)
3. [Tuples & Unpacking](#3-tuples--unpacking)
4. [Sets](#4-sets)
5. [Slicing & Indexing](#5-slicing--indexing)
6. [Comprehensions & Generator Expressions](#6-comprehensions-and-generator-expressions)
7. [Functions (including Lambdas)](#7-functions-including-lambdas)
8. [iterools & functools](#8-itertools--functools)
9. [Classes & DataClasses](#9-classes--dataclasses)
10. [Context Managers](#10-context-managers)
11. [Error Handling](#11-error-handling)
12. [Type Hints & the typing Module](#12-type-hints--the-typing-module)
13. [Virtual Environments & Dependency Management](#13-virtual-environments--dependency-management)
14. [Unit Testing (pytest)](#14-unit-testing-pytest)
15. [Logging](#15-logging)
16. [Performance Tips](#16-performance-tips)
17. [Debugging](#17-debugging)
18. [Jupyter Notebooks](#18-jupyter-notebooks)

---

## 1. Lists & List Comprehensions

**What:** Lists are used in data science to store collections of data, such as rows in a dataset or individual feature values.

**Why:** List comprehensions provide a clear and compact way to create or transform these lists, making data manipulation tasks faster and more readable—an essential part of data wrangling.

**Python lists vs. C-style arrays:** Unlike C arrays, Python lists can store elements of any type and can grow or shrink dynamically. For large numerical arrays, consider using the `array` module (for numbers) or, more commonly in data science, `numpy` arrays for efficiency and extra functionality.

**Example:**

```python
temperatures_c = [12, 18, 22, 15]
temperatures_f = [c * 9/5 + 32 for c in temperatures_c] # List comprehension
print(temperatures_f)
```

**You should be able to:**  
- Create, access, and modify lists  
- Write and read list comprehensions (including nested comprehensions)  
- Recognize when to use lists versus NumPy arrays for large, homogeneous data

---

## 2. Dictionaries & Nested Dicts

**What:** Dicts map keys to values—ideal for label-value pairs, config settings, and JSON-like data.

**Why:** Nested dicts are common when representing structured or hierarchical information.

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

**What:** Tuples are immutable sequences — once created, their contents cannot be changed.

**Why:** They’re ideal for fixed-size, heterogeneous data (e.g. coordinate pairs, function returns). Tuples support multiple assignment and “starred unpacking” for flexible splitting.

**Tuple immutability:**  
Once a tuple is created, you cannot change its contents:
```python
# no-run
point = (2, 3)
# point[0] = 5 would raise a TypeError since you cannot edit tuples
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

**What:** Sets store unique items; great for deduplication, membership testing, and set algebra (union/intersection).

**Why:** Used often for label sets, removing duplicates, or fast lookups.

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

**What:** Slicing extracts sublists or substrings—vital for data selection and windowing.

**Multi-level slicing:**  
You can slice lists of lists (matrices) or strings:

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
row = matrix[1]      # row = [4, 5, 6]
submatrix = [r[1:] for r in matrix[:2]]  # submatrix = [[2, 3], [5, 6]]

text = "data science"
print(text[5:12])
```

**Example:**

```python
# no-run
data = [0, 1, 2, 3, 4, 5]
window = data[2:5] # window = [2, 3, 4]
reversed_data = data[::-1] # reversed_data = [5, 4, 3, 2, 1]
```

**You should be able to:**  
- Use slicing for lists, strings, and (with libraries) NumPy arrays  
- Read and write multi-level slices  
- Understand step and negative indexing

---

## 6. Comprehensions and Generator Expressions

**What:** Comprehensions (list/dict/set) eagerly build collections in memory; generator expressions produce items lazily.

**Why:** Crucial for efficiency when dealing with large or streaming data sets.

**Memory usage & lazy evaluation:**  
- List comprehensions store the entire result in memory—fast for small/medium data, potentially inefficient for huge data.
- Generator expressions yield one result at a time using lazy evaluation—consume less memory, suitable for pipelines or iterating over large data sets.

**When to use:**  
- Use comprehensions for transformations where you need the entire result at once.
- Use generators when you process data one item at a time, or the result would be too large to fit in memory.

**Example:**

```python
# no-run
squares_comprehension = [x**2 for x in range(10)]
squares_generator = (x**2 for x in range(10))
```

**You should be able to:**  
- Choose between comprehensions and generators based on data size and memory constraints  
- Write both styles for data wrangling tasks  
- Recognize generator exhaustion

---

## 7. Functions (including Lambdas)

**What:** Functions organize code and logic; lambdas create small, anonymous functions.

**Why:** Handy for sorting, filtering, or passing as arguments.

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
```

**Lambda functions:**
```python
data = ["apple", "pear", "banana"]
data.sort(key=lambda word: len(word))
print(data)
```

**You should be able to:**  
- Define and call functions with positional, keyword, and default arguments  
- Use lambdas in appropriate contexts  
- Pass functions as arguments (first-class functions)

---

## 8. itertools & functools

**What:** These standard library modules provide advanced iteration and functional utilities (e.g., grouping, mapping, accumulating).

- `itertools.chain`: concatenate iterables  
- `itertools.product`: Cartesian product  
- `itertools.combinations`: all possible pairs/groups  
- `itertools.groupby`: group consecutive items  
- `functools.reduce`: accumulate a result  
- `functools.partial`: pre-fill function args

**Examples:**

```python
from functools import reduce

numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print(product)
```

```python
from itertools import groupby
data = ["a", "aa", "b", "bb", "b"]
groups = {k: list(g) for k, g in groupby(sorted(data), key=lambda x: x[0])}
print(groups)
```

**You should be able to:**  
- Use chain, product, combinations, groupby for iteration tasks  
- Use reduce and partial for custom accumulation/function manipulation  
- Recognize when built-in alternatives are preferable (e.g., sum/min/max)

---

## 9. Classes & DataClasses

**What:** *Classes* encapsulate data and methods, supporting inheritance and customization.

**Why:** *DataClasses* (Python 3.7+) are specialized classes for simple data containers, generating boilerplate (e.g., `__init__`, `__repr__`) automatically.

Below are two equivalent ways to define a simple data container: a classic class and a dataclass.

### Classic class example

```python
class Measurement:
    def __init__(self, id, value):
        self.id = id
        self.value = value

m = Measurement(1, 3.2)
print(m.value)
```

### Dataclass equivalent

```python
# no-run
from dataclasses import dataclass

@dataclass
class Measurement:
    id: int
    value: float

m = Measurement(1, 3.2)
```

**You should be able to:**  
- Write and use custom classes for encapsulating data/behavior  
- Use dataclasses for simple record types  
- Decide when to use each based on project needs

---

## 10. Context Managers

**What:** Use `with` to manage resources (files, DB connections) safely—ensures cleanup even on error.

**How it works:**  
Context managers implement the `__enter__` and `__exit__` methods, which set up and tear down resources automatically. You can write them as classes or using `contextlib.contextmanager`.

**Example:**

```python
# no-run
with open('results.txt', 'w') as f:
    f.write("Experiment complete.") # This will overwrite the contents of results.txt with one line: Experiment complete
```

**Custom context manager with contextlib:**

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(task_name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{task_name} finished in {elapsed:.4f}s")

# Usage
with timer("Heavy computation"):
    total = sum(i * i for i in range(10_000_000))
```

*This context manager measures execution time for any block and prints the duration—handy for quick performance checks.*

**You should be able to:**  
- Use `with` for file/database/network resource management  
- Recognize and implement custom context managers (class or decorator style)  
- Understand the role of `__enter__` and `__exit__`

---

## 11. Error Handling

**What:** Proper use of `try`/`except` ensures robust pipelines, especially with unpredictable data.

**Catching specific exceptions, raising new ones, and using all blocks:**

```python
try:
    val = float("not a number")
except ValueError as e:
    print(f"Conversion failed: {e}")
    # raise RuntimeError("Parsing error") from e
else:
    print("Conversion succeeded!")
finally:
    print("This will print no matter what!")
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

**What:** Type hints clarify intent, improve code readability, and enable static analysis tools (e.g., mypy, IDEs).

**Why:** The `typing` module offers generics (e.g., `List[T]`, `Dict[K, V]`), `Optional`, `Union`, and more.
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

print(mean([1.0, 2.0, 4.0]))
```

**You should be able to:**  
- Add type hints to functions, classes, and variables  
- Use common generics (`List`, `Dict`, `Optional`, `Union`) and TypedDict for structured data  
- Run static type checks with mypy or similar tools

---

## 13. Virtual Environments & Dependency Management

**What:** Isolating environments prevents package conflicts and ensures reproducibility.

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

**What:** Tests catch regressions and document expectations.

**Why:** `pytest` is the de facto testing library.

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
    assert result == 5 ## if result is not equal to 5 (i.e. the add function does not work as intended) then this will raise an error
```

- Fixtures in pytest help you set up reusable test data and environments.

**You should be able to:**  
- Write simple unit tests using pytest  
- Understand arrange/act/assert structure  
- Use and recognize pytest fixtures

---

## 15. Logging

**What:** Logging (vs. print) enables scalable monitoring, diagnostics, and debugging.

**Why:** Logging supports different levels (INFO, WARNING, ERROR), structured logs, and integrates with ML pipeline tools and experiment tracking.

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

## 16. Performance Tips

**Vectorization:**  
Perform operations on whole collections (arrays) at once using numpy/pandas. This avoids slow Python loops and leverages fast, compiled code underneath.

```python
import numpy as np
a = np.arange(1000000)
b = a * 2  # vectorized and therefore fast
```
*Why crucial:* Vectorization massively speeds up code, especially with large datasets.

**Generators:**  
Generators yield one item at a time, allowing you to process data streams or large files without loading everything into memory.

```python
# no-run
def gen():
    for i in range(1_000_000):
        yield i**2

for val in gen():
    if val > 100:
        break # this will break when i=11, val=121 and not calculate val for any more values of i
```
*Why crucial:* Prevents memory overload and enables pipeline-style processing.

**Memory Views:**  
Memory views allow you to work efficiently with large binary data (e.g., images or raw byte buffers), avoiding unnecessary copying.

```python
buf = bytearray(b"abcdefgh")
view = memoryview(buf)
print(view[2:5].tobytes())
```
*Why crucial:* Enables fast, zero-copy operations on large datasets.

**You should be able to:**  
- Use vectorized operations for numerical data  
- Write and consume generators for large/streamed data  
- Utilize memoryview for advanced binary data tasks

---

## 17. Debugging

**What:** Debuggers (`pdb`, IDE tools) and assert statements are invaluable for inspecting data flows and catching subtle bugs.

**You should be able to:**  
- Use assert for sanity checks  
- Set breakpoints and step through code  
- Use pdb or IDE debuggers for troubleshooting

---

## 18. Jupyter Notebooks

Jupyter Notebooks are powerful, interactive documents that combine live code, visualizations, and explanatory text. Widely used in data science, teaching, and research, notebooks allow you to interleave executable code (Python or other languages), formatted notes (Markdown), equations, and rich outputs (plots, tables, images) in a single, shareable file. Each notebook consists of "cells"—the two most common types are **code cells** (which you run to produce output) and **Markdown cells** (for formatted text, math, or instructions).

When working in a notebook, you execute code cells individually, and the results—such as printed output, plots, or error messages—appear immediately below the cell. The underlying process that runs your code is called a **kernel**, which maintains the execution state and variables across cells (so earlier results can be reused later). Restarting the kernel clears this state. Notebooks are ideal for exploratory data analysis (EDA), rapid prototyping, teaching, and sharing reproducible reports or data workflows.

---

# Key Takeaways

- Data science with Python builds on solid core language skills
- Mastering lists, dicts, comprehensions, generators, and classes is essential for readable, high-performance code.
- Key ecosystem libraries like NumPy, pandas, Matplotlib, and scikit-learn are foundational—learn their idioms and APIs.
- Robust workflow includes using virtual environments, testing, logging, and performance-aware coding.

---

# Exercise
"""
Generate a list `squares` containing the squares of numbers 1-10 using a list comprehension, then print the result.
"""
```python
squares = [n**2 for n in range(1, 11)]
print(squares)
```