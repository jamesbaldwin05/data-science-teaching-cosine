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
    - [Importing NumPy](#importing-numpy)
    - [Creating an array](#creating-an-array)
    - [Array Data Types](#array-data-types)
    - [Indexing & Slicing](#indexing--slicing)
2. [Array Attributes](#array-attributes)
    - [Array Properties](#array-properties)
    - [Reshaping Arrays](#reshaping-arrays)
    - [Changing Dimensions](#changing-dimensions)

---

## NumPy Basics
NumPy arrays are powerful tools designed for efficient numerical computing. Unlike regular Python lists, which can hold mixed data types and are slower for math operations, NumPy arrays store elements of the same type in a compact way and support fast, element-wise calculations without needing loops. They also handle multi-dimensional data and offer many built-in mathematical functions, making them ideal for scientific and data-intensive tasks.

### Importing NumPy
To import the NumPy library, use the code below:
```python
# no-run
import numpy as np
```
It is de-facto standard to use `np` as an alias for NumPy.

Every example beyond this point will not show the code to import but it still needs to be there to work correctly.

### Creating an array
To create a NumPy array from a list, you can call `np.array`.

```python
arr = np.array([1, 2, 3])
print(arr)
```

There are also a number of built in generators:

- `np.zeros(shape)` creates an array filled with zeros.

```python
arr = np.zeros((3,3))
print(arr)
```

- `np.ones(shape)` creates an array filled with ones.

```python
arr = np.ones(6)
print(arr)
```

- `np.full(shape, fill_value)` creates an array filled with a given value.

```python
arr = np.full((2,3), 7)
print(arr)
```

- `np.eye(n)` creates an identity matrix of size n x n.

```python
arr = np.eye(4)
print(arr)
```

- `np.arange(start, stop, step)` creates an array with evenly spaced values within a range (start inclusive, stop exclusive).

```python
arr = np.arange(1,10) # default step is 1
print(arr)
```

- `np.linspace(start, stop, num)` creates an array with the specified number of values evenly spaced between start and stop (both inclusive).

```python
arr = np.linspace(0, 1, 11)
print(arr)
```

### Array Data Types
When using `np.array()` the data type of the array is inferred from the list. However, the default data types for generators (such as `np.zeros()` or   `np.full()`) is a `float` (you may have already noticed this in previous examples). To change this data type, we need to pass `dtype` as an argument.

```python
arr = np.zeros(5, dtype=int)
print(arr)
```

To change an arrays data type once it has been defined, we can use the `astype()` method.

```python
arr = np.zeros(5)         # [0. 0. 0. 0. 0.]
arr = arr.astype(int)
print(arr)
```

### Indexing & Slicing

NumPy arrays support multiple ways to access and manipulate elements, similar to python lists:

- **1D & 2D slicing**: use `start:stop:step` syntax.  
- **Negative indexing**: counts from the end.  

```python
# no-run
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# 2D slicing
arr[0:2, 1:3]
# [[2 3]
#  [5 6]]

# Negative indexing
arr[-1, -2]  # 8
```

They also support other indexing methods (which would need a list comprehension with native python lists):

- **Fancy indexing**: selects multiple specific elements using a list of indices.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

values = arr[[0, 1], [0, 2]]   # select elements at (0,0) and (1,2)
print(values)

rows = arr[[0, 2], :]             # select rows 0 and 2
print(rows)
```

- **Boolean indexing**: select elements based on a condition, returning them in a 1D array.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

condition = arr[arr > 5]
print(condition)
```

---

## Array Attributes

### Array Properties

- `.shape` returns a tuple representing the dimensions of the array.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)
```

- `.ndim` returns the number of dimensions (axes) in the array.

```python
arr = np.array([[[1, 1], [2, 2], [3, 3]]])
print(arr.ndim)
```

- `.size` returns the total number of elements in the array.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.size)
```

- `.dtype` returns the data type of the elements.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.dtype)
```

- `.itemsize` returns the size (in bytes) of each element.
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.itemsize)
```

### Reshaping Arrays

- `.reshape(new_shape)` returns a new view of the array with the given shape (must have the same number of total elements).

```python
arr = np.arange(6)
reshaped = arr.reshape((2, 3))
print(reshaped)
```

- `.ravel()` and `.flatten()` both change an array to 1D (if possible) but `.ravel()` returns a view of the array in 1D, modifying the contents will change the original whereas `.flatten()` returns a copy of the array in 1D, modifying the contents will not change the original but it is slower.

```python
arr = np.array([[1, 2], [3, 4]])

ravel = arr.ravel()               #[1 2 3 4]
ravel[0] = 99
print(arr)                        # original array is changed

arr = np.array([[1, 2], [3, 4]])
flatten = arr.flatten()           # [1 2 3 4]
flatten[1] = 100
print(arr)                        # original array is unchanged
```

### Changing Dimensions

- `np.newaxis` adds a new axis, increasing the number of dimensions by 1.

```python
arr = np.array([1, 2, 3])
arr = arr[:, np.newaxis]

print(arr)
print(arr.shape)
```

- `np.expand_dims(a. axis)` adds a new axis at a specified position, increasing the number of dimensions by 1.

```python
arr = np.array([1, 2, 3])
expanded0 = np.expand_dims(arr, axis=0)  # adds axis at position 0
print(expanded0)

expanded1 = np.expand_dims(arr, axis=1)  # adds axis at position 1
print(expanded1)
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