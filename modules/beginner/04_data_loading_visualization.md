# Data Loading & Visualization

**Concept**  
Reading data from CSV or Excel is the first step. Use `pandas.read_csv()` or `read_excel()`. Visualization libraries like matplotlib let you plot data quickly.

### Example
```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
df["sepal_length"].hist()
plt.title("Sepal Length Distribution")
plt.xlabel("Length")
plt.ylabel("Count")
plt.show()
```

### Exercise
"""
Read the Titanic CSV and plot a histogram of passenger fares.
"""
```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** Which pandas function reads Excel files?
- A) pd.read_excel
- B) pd.open_excel
- C) pd.load_excel
- D) pd.read_csv
**Answer:** A