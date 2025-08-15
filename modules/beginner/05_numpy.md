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
3. [Basic Operations](#basic-operations)
    - [Elementwise Arithmetic](#elementwise-arithmetic)
    - [Elementwise Functions](#elementwise-functions)
    - [Comparison Operators](#comparison-operators)
    - [Aggregations](#aggregations)
4. [Random Numbers](#random-numbers)
    - [Random Distributions](#random-distributions)
    - [Shuffling](#shuffling)
5. [Broadcasting](#broadcasting)
6. [Advanced Indexing](#advanced-indexing)
    - [Boolean Indexing](#boolean-indexing)
    - [Fancy Indexing](#fancy-indexing)
    - [Conditional Selection](#conditional-selection)
7. [Linear Algebra with NumPy](#linear-algebra-with-numpy)

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

---

## Basic Operations

NumPy arrays support **vectorised operations** meaning operations are applied **elementwise**, without the need for loops.

### Elementwise Arithmetic

- `+` and `-` or `np.add()` and `np.subtract()` perform addition/subtraction elementwise.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 + arr2)            # or np.add(arr1, arr2)
print(np.subtract(arr1, 1))   # subtracts 1 from every element

```

- `*` and `/` or `np.multiply()` and `np.divide()` perform multiplication/division elementwise.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 * arr2)            # or np.multiply(arr1, arr2)
print(np.divide(arr1, 2))     # divides every element by 2

```

- `**` or `np.power()` raises each element to a power elementwise.

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 ** arr2)           # each element in arr1 is raised to the corresponding power in arr2
print(np.power(arr1, 2))      # squares every element
```

### Elementwise Functions
- `np.sqrt()` takes the square root of each element in the array

```python
arr = np.array([1, 2, 3])
print(np.sqrt(arr))
```

- `np.log()` takes the natural logarithm of each element in the array

```python
arr = np.array([1, 2, 3])
print(np.log(arr))
```

- `np.exp()` takes the exponential of each element in the array

```python
arr = np.array([1, 2, 3])
print(np.exp(arr))
```

### Comparison Operators
We can use comparison operators elementwise in NumPy, which will return an array of the same shape with `True` and `False` values.

```python
arr = np.array([[1, 5, 9],
                [4, 7, 2]])

print(arr > 4)
```

```python
arr = np.array([[1, 5, 9],
                [4, 7, 2]])

print(arr == 1)
```

It can be used with boolean indexing but this will flatten the array to 1D.

```python
arr = np.array([[1, 5, 9],
                [4, 7, 2]])

arr2 = arr[arr < 3]
print(arr2)
```

### Aggregations
These are a group of methods that compute a single value over the whole array

- `.sum()` returns the sum of every element in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.sum())
```

- `.mean()` returns the mean of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.mean())
```

- `.max()` returns the maximum value in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.max())
```

- `.min()` returns the minimum value in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.min())
```

- `.std()` returns the standard deviation of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.std())
```

- `.var()` returns the variance of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.var())
```

## Random Numbers
NumPy’s `np.random` module provides tools for generating random numbers, sampling from distributions, and shuffling data.

### Random Distributions

- `np.random.rand(shape)` fills an array of the given shape with random floats in [0, 1]

```python
print(np.random.rand(2,2)) 
```

- `np.random.randn(shape)` fills an array of the given shape with random samples from a standardised normal distribution (mean 0, std 1)

```python
print(np.random.randn(2,2,2)) 
```

- `np.random.randint(low, high, size)` fills an array of the given shape (`size`) with random integers between low (inclusive) and high (exclusive) 

```python
print(np.random.randint(1, 10, (2,2)))
```

- `np.random.seed(value)` sets a seed for the random generation so results are reproducible. Each integer will give a constant set of random numbers.

```python
np.random.seed(0)
print(np.random.rand(2,2))
```

- `np.random.uniform(low, high, size)` fills an array of the given shape (`size`) with random samples from a given uniform distribution (between low and high).

```python
print(np.random.uniform(100, 120, (2,2)))
```

- `np.random.normal(mean, std, size)` fills an array of the given shape (`size`) with random samples from a given normal (Gaussian) distribution.

```python
print(np.random.normal(5, 2, (2,2)))
```

### Shuffling

- `np.random.shuffle(arr)` shuffles an array **in place**. It has no return value and directly affects the original array.

```python
arr = np.array([1, 2, 3, 4])
np.random.shuffle(arr)
print(arr)
```

- `np.random.permutation()` returns a new shuffled array, leaving the original unchanged.

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.random.permutation(arr1)
print(arr1)
print(arr2)
```

## Broadcasting
Broadcasting is NumPy’s way of performing elementwise operations on arrays with **different shapes** by virtually expanding (stretching) arrays with a dimension of 1 to match the other array, without actually copying data. This allows operations that standard linear algebra rules wouldn’t normally permit, avoiding explicit loops and making calculations faster and more memory-efficient.

Rules:
- Compare shapes, starting from the trailing dimensions.
```less
Suppose we have A with shape (2, 3, 4) and B with shape (3, 1)
A:    2   3   4
B:        3   1

Missing leading dimension in B is treated as 1.
A:    2   3   4
B:   (1)  3   1
```

- Dimensions must be equal, or one of them must be equal to 1.
- If one dimension is 1, it is "stretched" to match the other.
```less
From the right,
A:4, B:1 is allowed with B stretched to 4.
A:3, B:3 is allowed.
A:2, B:1 is allowed with B stretched to 2.

If B had shape (2,1) (for example), it would not be allowed
```

Examples of broadcasting:
- Array * Scalar

```python
arr = np.array([[1, 2, 3],
                  [4, 5, 6]])

print(arr*10)
```

- 1D array + 2D array

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
arr1d = np.array([10, 20, 30])

print(arr2d + arr1d)
```

- 2D array + 2D array

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[10],
              [20]])

print(A + B)
```

---

## Advanced Indexing

NumPy also supports more advanced indexing methods, such as boolean indexing and fancy indexing, which would require more verbose approaches like list comprehensions with native Python lists.

### Boolean Indexing
Selects elements from an array based on a condition, returning a 1D array of the elements that satisfy it.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

condition = arr[arr > 5]
print(condition)
```

### Fancy Indexing
Selects elements from an array using a list or array of indices, allowing arbitrary positions to be picked.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

values = arr[[0, 1], [0, 2]]   # select elements at (0,0) and (1,2)
print(values)

rows = arr[[0, 2], :]             # select rows 0 and 2
print(rows)
```

We can also combine boolean and fancy indexing:

```python
arr = np.array([10, 15, 20, 25, 30])

mask = arr > 15                        # [False, False, True, True, True]

print(arr[mask][[0, 2]])               # select the elements at indexes 0 and 2 from the "True" part of the mask, this does not affect the original array
```

### Conditional Selection
Using `np.where` we can do conditional selection with NumPy:
`np.where(condition, x, y)` selects elements from two arrays (or values) based on a condition, producing a new array where each element is:
- `x` if the condition is `True`
- `y` if the condition is `False`

If `x` and `y` are omitted, `np.where(condition)` returns indices where the condition is true.

```python
arr = np.array([1, 5, 10, 15, 100, 1000])

result = np.where(arr % 2 == 0, arr*2, arr*3)     # all even numbers are multiplied by 2, all odd numbers are multiplied by 3
print(result)

indices = np.where(arr>10)                        # returns the indices where this condition is true
print(indices[0])                                 # np.where returns a tuple of arrays, one per dimension; [0] gets the 1D indices
```

---

## Linear Algebra with NumPy
NumPy provides various tools for standard linear algebra operations via `np` and `np.linalg`.

- **Dot product**:  
`np.dot(A, B)` returns the sum of elementwise products.

```python
A = np.array([1, 5, 2])
B = np.array([9, 3, 4])

print(np.dot(A, B))         # A @ B will also return the dot product if two 1D arrays are passed
```

- **Matrix Multiplication**:  
For 2D arrays, `A @ B` will work out the matrix multiplication of two arrays. For higher dimensional arrays, `np.matmul(A, B)` is more explicit.

```python
A = np.array([[2, 4],
              [5, 1]])

B = np.array([[7, 3],
              [3, 8]])

print(A @ B)                # np.dot(A, B) will also return the matrix multiplication if two 2D arrays are passed.
```

- **Transpose**:  
`.T` will return the transpose of the array.

```python
A = np.array([[2, 4],
              [5, 1]])

print(A.T)
```

- **Determinant**:  
`np.linalg.det(A)` will return the determinant of a square array.

```python
A = np.array([[3, 1],
              [3, 1]])

print(np.linalg.det(A))
```

- **Inverse**:  
`np.linalg.inv(A)` will return the inverse of a square array (if invertible).

```python
A = np.array([[2, 4],
              [5, 1]])

print(np.linalg.inv(A))
```

- **Solve Linear Systems**:
`np.linalg.solve(A, b)` will solve the linear system $Ax = b$.

```python
A = np.array([[2, 4],
              [5, 1]])

b = np.array([9,8])

print(np.linalg.solve(A, b))
```

- **Eigenvalues & Eigenvectors**:  
`np.linalg.eig(A)` will return the eigenvalues and eigenvectors (in a tuple) of a square array.

```python
A = np.array([[2, 4],
              [5, 1]])

evalues, evectors = np.linalg.eig(A)

print(evalues)
print(evectors)                        # eigenvectors are the columns of this array
```

- **Norms**:  
`np.linalg.norm(v, ord)` will return the length/size of vectors or magnitude of matrices from one of various norms.

```python
v = np.array([3, -4])

A = np.array([[2, 4],
              [5, 1]])

print("Euclidean/L2 norm", np.linalg.norm(v))                               # = sqrt(3^2 + (-4)^2)
print("Manhattan/L1 norm", np.linalg.norm(v, 1))                            # = |3| + |-4|
print("Infinity norm (max absolute value):", np.linalg.norm(v, np.inf) )    # = max(|3|, |-4|)
print("Frobenius norm/L2 norm for matrices:", np.linalg.norm(A))            # = sqrt(2^2 + 4^2 + 5^2 + 1^2)
```

- **SVD (Single Value Decomposition)**:  
`np.linalg.svd(A)` will return (as a tuple) an array decomposed into the form $U \Sigma V^T$.

```python
A = np.array([[1, 2],
              [3, 4], 
              [5, 6]])
U, S, Vt = np.linalg.svd(A)
print(U)
print(S)
print(Vt)
```

- **Pseudoinverse**:  
`np.linalg.pinv(A)` will return the pseudoinverse of A.

```python
A = np.array([[1, 2],
              [3, 4], 
              [5, 6]])
print(np.linalg.pinv(A))
```

---