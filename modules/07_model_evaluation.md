# Evaluating Models

**Concept**  
After training a model, you must evaluate how well it works. Use metrics like accuracy (for classification), mean squared error (for regression), or confusion matrix for detailed results. Always test on data not seen during training.

### Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
X = df[["age", "fare"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
```

### Exercise
"""
Compute and print the confusion matrix for your Titanic survival model.
"""
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "titanic.csv")
df = df.dropna(subset=["age"])
X = df[["age", "fare"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Your code here:
```

### Quiz
**Question:** Which metric shows both true/false positives and negatives for classification?
- A) accuracy_score
- B) confusion_matrix
- C) mean_squared_error
- D) roc_curve
**Answer:** B