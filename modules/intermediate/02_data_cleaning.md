# Data Cleaning

**Concept**  
Data cleaning prepares raw data for analysis. Common steps: handling missing values (`df.isnull()`), removing duplicates, and detecting outliers. Clean data improves model accuracy.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
print("Missing ages:", df["age"].isnull().sum())
df_clean = df.dropna(subset=["age"])  # Remove rows with missing age
print("Rows after cleaning:", len(df_clean))
```

### Exercise
"""
Remove duplicate rows from the Titanic dataset and print the new shape.
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Q1:** Which pandas method removes duplicate rows?
- A) df.clean()
- B) df.drop_duplicates()
- C) df.remove()
- D) df.unique()
**A:** B

**Q2:** What value does pandas use to represent missing data?
**A:** NaN