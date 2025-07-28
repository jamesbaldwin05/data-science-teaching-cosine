# 04 Supervised Learning

**Concept**  
Supervised learning uses labeled data to train models for classification (predicting categories) or regression (predicting numbers). Algorithms include logistic regression, decision trees, and random forests. Always split data into train/test sets.

### Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
X = df[["age", "fare"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Train accuracy:", model.score(X_train, y_train))
```

### Exercise
"""
Train a logistic regression using "age" and "fare" to predict survival. Print test accuracy.
"""
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
X = df[["age", "fare"]]
y = df["survived"]
# Your code here:
```

### Quiz
**Question:** Which library provides ready-to-use ML models in Python?
- A) matplotlib
- B) scikit-learn
- C) seaborn
- D) tensorflow
**Answer:** B