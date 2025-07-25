# Getting Started with Data

**Concept**  
Data science is about finding meaning in data using code. As a developer, you already know how to work with files and code. In data science, we load datasets (often CSV files), inspect them, and manipulate them using libraries like `pandas`. Let's see how to load and view a dataset.

### Example
```python
import pandas as pd
from pathlib import Path

# Load the Iris dataset from the local data folder
data_path = Path(__file__).resolve().parent.parent / "data" / "iris.csv"
df = pd.read_csv(data_path)
print(df.head())  # Show first 5 rows
```

### Exercise
"""
Load the Titanic dataset (data/titanic.csv) into a DataFrame and print the first 3 rows.
"""
```python
import pandas as pd
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
# Your code here:
```

### Quiz
**Question:** Which Python library is most commonly used to work with tabular data?
- A) numpy
- B) pandas
- C) matplotlib
- D) seaborn
**Answer:** B