# 06 Unsupervised Learning

**Concept**  
Unsupervised learning finds patterns in data without labels. Clustering (e.g., k-means) groups similar items; dimensionality reduction (PCA) simplifies features.

### Example
```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "iris.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print("Cluster assignments:", labels[:10])
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
print("PCA shape:", X2.shape)
```

### Exercise
"""
Run KMeans clustering on the Titanic 'age' and 'fare' columns, and print the cluster labels.
"""
```python
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "titanic.csv")
X = df[["age", "fare"]].dropna()
# Your code here:
```

### Quiz
**Question:** Which technique reduces a dataset to fewer dimensions while preserving variance?
- A) Clustering
- B) PCA
- C) Regression
- D) Filtering
**Answer:** B