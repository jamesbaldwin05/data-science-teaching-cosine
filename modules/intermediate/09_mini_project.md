# 09 Mini Project: Titanic Survival

**Concept**  
Apply your skills: load Titanic data, engineer features, train and evaluate a model. This end-to-end process mirrors real-world data science.

### Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
df["age_group"] = df["age"].apply(lambda x: "child" if x < 18 else "adult")
X = pd.get_dummies(df[["age", "fare", "sex", "age_group"]])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.2f}")
```

### Exercise
"""
Add 'pclass' (passenger class) as a feature to the model. What happens to test accuracy?
"""
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
# Your code here:
```

### Quiz
**Q1:** What is a typical first step in a real-world ML project?
- A) Train a model immediately
- B) Explore and clean the data
- C) Export results to Excel
- D) Tune hyperparameters
**A:** B

**Q2:** True or False: Feature engineering often comes after initial data exploration.
**A:** True