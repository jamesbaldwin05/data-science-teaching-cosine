# Advanced Visualization

**Concept**  
Advanced visualizations use color, style, and interactivity. Customize matplotlib/seaborn plots with labels, legends, color palettes, and subplots for more insight.

### Example
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species", style="species")
plt.title("Iris Sepal vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()
plt.show()
```

### Exercise
"""
Make a seaborn boxplot of Titanic fares, colored by 'pclass'.
"""
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
# Your code here:
```

### Quiz
**Q1:** Which function sets a color palette in seaborn?
- A) sns.set_palette
- B) plt.set_palette
- C) sns.palette
- D) plt.set_color
**A:** A

**Q2:** Name a function to display a legend in matplotlib.
**A:** legend