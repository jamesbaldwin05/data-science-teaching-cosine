### Broadcasting
- Broadcasting in NumPy is a way to perform operations on arrays of different shapes by automatically expanding the smaller array to match the shape of the larger one without actually copying data. For example, adding a scalar to a matrix adds the scalar to every element.

- This is **not** mathematically valid in strict linear algebra because operations like addition are only defined for arrays (or matrices/vectors) of the same shape. Broadcasting relaxes that rule for programming convenience, but it's a computational shortcut, not a formal mathematical operation. Broadcasting corresponds to implicitly repeating a vector across rows or columns as needed for the operation (a programming convenience), even though classical linear algebra only defines addition for same-shaped arrays.


```python
import numpy as np

# Add a vector to each row of a matrix
B = np.array([[1, 2], [3, 4], [5, 6]])
v = np.array([10, 100])
print(B + v)
```

---

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
import numpy as np
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