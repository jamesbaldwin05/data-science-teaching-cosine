# DataFrames with pandas

**Concept**  
A DataFrame is like a table: rows are records, columns are features. `pandas` lets you select, filter, and modify data with simple code. You can access columns, filter rows by conditions, and compute new values easily.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "iris.csv")
# Select a column
print(df["species"].unique())
# Filter rows
setosa = df[df["species"] == "setosa"]
print(setosa.head())
```

### Exercise
"""
Select all Titanic passengers aged under 18, and print their names and ages.
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** What function reads a CSV file into a DataFrame?
- A) pd.read_table
- B) pd.read_csv
- C) pd.open_csv
- D) pd.import_csv
**Answer:** B