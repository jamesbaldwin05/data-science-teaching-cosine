# Data Exploration

**Concept**  
Explore your data before modeling: check shapes, types, missing values, summary stats. Use `df.info()`, `df.describe()`, and value counts for quick insights.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
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

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** What method gives summary statistics (mean, std, etc.) for numeric columns in a DataFrame?
- A) df.info()
- B) df.describe()
- C) df.head()
- D) df.value_counts()
**Answer:** B