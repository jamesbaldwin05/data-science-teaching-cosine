# NumPy (Numerical Python)
NumPy (Numerical Python) is a fundamental Python library used for numerical computing and data manipulation. It provides a powerful, flexible, n-dimensional array object `ndarray` that allows efficient storage and operations on large datasets of homogeneous data (typically numbers). It allows for:
- Fast, vectorised operations on arrays due to elementwise maths without slow python loops.
- A large set of mathematical functions including linear algebra, statistics, Fourier transforms and random number generation.
- Tools for reshaping, indexing and slicing data
- Efficient memory management
- Seamless integration with many other scientific libraries including Pandas, SciPy and Matplotlib

---

## Table of Contents
1. [NumPy Basics](#numpy-basics)

---

## NumPy Basics
NumPy arrays are powerful tools designed for efficient numerical computing. Unlike regular Python lists, which can hold mixed data types and are slower for math operations, NumPy arrays store elements of the same type in a compact way and support fast, element-wise calculations without needing loops. They also handle multi-dimensional data and offer many built-in mathematical functions, making them ideal for scientific and data-intensive tasks.

To import the NumPy library, use the code below:
```python
import numpy as np
```
It is de-facto standard to use `np` as a alias for numpy. Every example beyond this point will not show this code but it still needs to be there to work correctly.

```python
# Hidden import
import numpy as np
```

### Creating an array

- From a list:  
```python
arr = np.array([1, 2, 3])
print(arr)
```





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