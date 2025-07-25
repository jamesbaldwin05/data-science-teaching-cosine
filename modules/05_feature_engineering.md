# Intro to Feature Engineering

**Concept**  
Feature engineering creates new variables from raw data to help models learn better. This can include scaling, encoding categories, or combining columns. For example, combining 'first_name' and 'last_name' into 'full_name'.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
# Create a new feature: "family_size"
df["family_size"] = df["sibsp"] + df["parch"] + 1
print(df[["name", "family_size"]].head())
```

### Exercise
"""
Add a new column "age_group" to Titanic data: "child" (<18), "adult" (18+).
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** What is feature engineering?
- A) Removing all missing data
- B) Creating new variables from raw data
- C) Training a machine learning model
- D) Changing file formats
**Answer:** B