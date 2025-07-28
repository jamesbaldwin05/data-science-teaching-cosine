# 05 Pandas Basics

**Concept**  
A DataFrame is a table of data. You can select, filter, and manipulate columns and rows easily. Pandas is the foundation for most data work in Python.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
print(df["species"].unique())
setosa = df[df["species"] == "setosa"]
print(setosa.head())
```

### Exercise
"""
Select all Titanic passengers aged under 18 and print their names and ages.
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** What function reads a CSV file into a DataFrame?
- A) pd.read_table
- B) pd.read_csv
- C) pd.open_csv
- D) pd.import_csv
**Answer:** B