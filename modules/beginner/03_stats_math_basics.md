# 03 Stats & Math Basics

**Concept**  
Descriptive statistics summarize data (mean, median, mode, std). Probability helps us reason about uncertainty. Visualizations like histograms and boxplots reveal distributions and outliers.

### Example
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
print("Mean sepal length:", df["sepal_length"].mean())
print("Median sepal length:", df["sepal_length"].median())
```

### Exercise
"""
Plot a boxplot of the 'age' column in the Titanic dataset.
"""
```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Question:** Which plot type best visualizes the spread and outliers of a variable?
- A) Scatter plot
- B) Boxplot
- C) Line chart
- D) Pie chart
**Answer:** B