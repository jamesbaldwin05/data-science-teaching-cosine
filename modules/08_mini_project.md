# Real-World Mini Project: Titanic Survival

**Concept**  
Let's apply what you've learned! We'll walk through loading the Titanic data, exploring features, building a model, and evaluating it. This end-to-end workflow is typical in real-world data science.

### Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
df["age_group"] = df["age"].apply(lambda x: "child" if x &lt; 18 else "adult")
X = pd.get_dummies(df[["age", "fare", "sex", "age_group"]])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
```

### Exercise
"""
Try adding 'pclass' (passenger class) as a feature to the model. What happens to test accuracy?
"""
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
# Your code here:
```

### Quiz
**Question:** What is a typical first step in a real-world ML project?
- A) Train a model immediately
- B) Explore and clean the data
- C) Export results to Excel
- D) Tune hyperparameters
**Answer:** B