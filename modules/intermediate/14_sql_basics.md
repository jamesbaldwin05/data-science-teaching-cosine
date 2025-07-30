# SQL Basics

**Concept**  
SQL is used to query and manipulate structured data. In data science, you often use SQL to pull data before analysis. With `sqlite3` you can run SQL on local files.

### Example
```python
import sqlite3
import pandas as pd

# Create in-memory DB and table
conn = sqlite3.connect(":memory:")
df = pd.DataFrame({"name": ["Ada", "Bob"], "age": [30, 40]})
df.to_sql("people", conn, index=False)
result = pd.read_sql_query("SELECT * FROM people WHERE age > 35", conn)
print(result)
```

### Exercise
"""
Write a SQL query to select all Titanic passengers with fare &gt; 50.
(Hint: use pandas.DataFrame.query() for this exercise.)
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Q1:** Which statement retrieves all rows from a table?
- A) GET * FROM table
- B) SELECT * FROM table
- C) FETCH * FROM table
- D) READ * FROM table
**A:** B

**Q2:** Which built-in Python module lets you use SQL on local files?
**A:** sqlite3