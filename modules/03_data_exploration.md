# Basic Data Exploration

**Concept**  
Before analyzing data, it's important to understand its structure. Data exploration means checking shapes, types, missing values, and summary statistics. `pandas` offers tools like `describe()` and `info()` to quickly summarize your dataset.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "iris.csv")
print(df.info())
print(df.describe())
```

### Exercise
"""
Check for missing values in the Titanic dataset. Print the count of missing values per column.
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** What method gives summary statistics (mean, std, etc.) for numeric columns in a DataFrame?
- A) df.info()
- B) df.describe()
- C) df.head()
- D) df.value_counts()
**Answer:** B