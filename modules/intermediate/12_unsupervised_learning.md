# Unsupervised Learning

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
**Q1:** Which technique reduces a dataset to fewer dimensions while preserving variance?
- A) Clustering
- B) PCA
- C) Regression
- D) Filtering
**A:** B

**Q2:** What is the goal of clustering?
**A:** group similar data


### Comparison operators

## PCA Example: Principal Component Analysis (Dimensionality Reduction)
- Principal Component Analysis (PCA) is a method for dimensionality reduction. It takes high-dimensional data and finds new axes (called principal components) that capture the most variance in the data and are uncorrelated (perpendicular to each other). It simplifies datasets while keeping most information and makes patterns easier to see.  

*More on variance in the statistics section.*

- Orthonormal sets (or orthonormal bases, though these do not mean the exact same thing) and diagonalizing matrices are heavily used in PCA.

- A simple example in 3D would be a set of data that lies almost in one plane (almost flat) with some tiny variation in one direction. PCA would find:  
  - PC1 - the direction of greatest variance (lying in the plane)
  - PC2 - the direction of second greatest variance (also lying in the plane)
  - PC3 - the direction of least variance (perpendicular to the plane and pointing "out" from it)  
- These components are all perpendicular to each other (they form an orthogonal/orthonormal basis).

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)
n = 20
x = np.random.normal(0, 1, n)
y = 1.5 * x + 0.5 + np.random.normal(0, 0.4, n)
data = np.column_stack((x, y))

pca = PCA(n_components=2)
pca.fit(data)
components = pca.components_                    #unit vectors
explained = pca.explained_variance_ratio_
mean = pca.mean_

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], color='skyblue', s=50, label='Data points')

plt.scatter(mean[0], mean[1], color='red', marker='x', s=80, label='Mean')

scale = 3
for i, (comp, var) in enumerate(zip(components, explained)):
    line = np.vstack([mean - comp * scale, mean + comp * scale])
    plt.plot(line[:, 0], line[:, 1],
             linewidth=2,
             label=f'PC{i+1} ({var:.2f} var)')

plt.axis('equal')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA on 2D Correlated Data')
plt.legend()
plt.show()
```

---