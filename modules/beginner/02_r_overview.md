# 02 R Overview for Data Science

**Concept**  
R is another popular language for data analysis and statistics, especially in academia. Its strengths are statistical modeling and visualization. Learn more at [R for Data Science](https://r4ds.had.co.nz/) and [CRAN documentation](https://cran.r-project.org/manuals.html).

### Example
```python
# Python's pandas is inspired by R's data.frame.
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
print(df.head())
```

### Exercise
"""
What is the equivalent of R's data.frame in Python? Print the type of 'df' after reading the CSV.
"""
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** Which R function is used to read CSV files?
- A) read.csv()
- B) open_csv()
- C) pd.read_csv()
- D) load.csv()
**Answer:** A