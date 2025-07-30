# Feature Engineering

**Concept**  
Feature engineering creates new variables from raw data to help models learn better: scaling, encoding categories, or combining columns. For example, combining 'sibsp' and 'parch' into 'family_size'.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
df["family_size"] = df["sibsp"] + df["parch"] + 1
print(df[["name", "family_size"]].head())
```

### Exercise
"""
Add a new column "age_group" to Titanic data: "child" (&lt;18), "adult" (18+).
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Q1:** What is feature engineering?
- A) Removing all missing data
- B) Creating new variables from raw data
- C) Training a machine learning model
- D) Changing file formats
**A:** B

**Q2:** True or False: Creating a new column by combining two features is feature engineering.
**A:** True