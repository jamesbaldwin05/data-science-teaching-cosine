# NumPy (Numerical Python)
NumPy (Numerical Python) is a fundamental Python library used for numerical computing and data manipulation. It provides a powerful, flexible, n-dimensional array object `ndarray` that allows efficient storage and operations on large datasets of homogeneous data (typically numbers). It allows for:
- Fast, vectorized operations on arrays due to elementwise maths without slow Python loops.
- A large set of mathematical functions including linear algebra, statistics, Fourier transforms, and random number generation.
- Tools for reshaping, indexing, and slicing data.
- Efficient memory management.
- Seamless integration with many other scientific libraries including Pandas, SciPy, and Matplotlib.

---

## Table of Contents
1. [NumPy Basics](#numpy-basics)
    - [Importing NumPy](#importing-numpy)
    - [Creating an Array](#creating-an-array)
    - [Array Data Types](#array-data-types)
    - [Indexing & Slicing](#indexing--slicing)
    - [Views vs Copies](#views-vs-copies)
2. [Array Attributes](#array-attributes)
    - [Array Properties](#array-properties)
    - [Reshaping Arrays](#reshaping-arrays)
    - [Changing Dimensions](#changing-dimensions)
    - [Stacking Arrays](#stacking-arrays)
    - [Splitting Arrays](#splitting-arrays)
3. [Basic Operations](#basic-operations)
    - [Elementwise Arithmetic](#elementwise-arithmetic)
    - [Elementwise Functions](#elementwise-functions)
    - [Comparison Operators](#comparison-operators)
    - [Aggregations](#aggregations)
    - [Reductions with Axis and Keepdims](#reductions-with-axis-and-keepdims)
    - [Arg and Index Utilities](#arg-and-index-utilities)
    - [NaN-aware Statistics](#nan-aware-statistics)
    - [Universal Function Parameters](#universal-function-parameters)
4. [Random Numbers](#random-numbers)
    - [NumPy Random Distributions](#numpy-random-distributions)
    - [Random Numbers using a Generator Object](#random-numbers-using-a-generator-object)
    - [Modern RNG Extras](#modern-rng-extras)
    - [Shuffling](#shuffling)
5. [Broadcasting](#broadcasting)
6. [Advanced Indexing](#advanced-indexing)
    - [Boolean Indexing](#boolean-indexing)
    - [Fancy Indexing](#fancy-indexing)
    - [Conditional Selection](#conditional-selection)
7. [Linear Algebra with NumPy](#linear-algebra-with-numpy)
8. [Statistics with NumPy](#statistics-with-numpy)
    - [Single Variable Statistics](#single-variable-statistics)
    - [Multivariate Statistics](#multivariate-statistics)
9. [Saving and Loading Data](#saving-and-loading-data)
10. [Polynomials](#polynomials)

---

## NumPy Basics
NumPy arrays are powerful tools designed for efficient numerical computing. Unlike regular Python lists, which can hold mixed data types and are slower for math operations, NumPy arrays store elements of the same type in a compact way and support fast, elementwise calculations without needing loops. They also handle multi-dimensional data and offer many built-in mathematical functions, making them ideal for scientific and data-intensive tasks.

### Importing NumPy
To import the NumPy library, use the code below:
```python
# no-run
import numpy as np
```
It is de facto standard to use `np` as an alias for NumPy.

Every example beyond this point will not show the code to import but it still needs to be there to work correctly.

### Creating an Array
To create a NumPy array from a list, you can call `np.array`.

```python
arr = np.array([1, 2, 3])
print(arr)
```

There are also a number of built-in generators:

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
When using `np.array()` the data type of the array is inferred from the list. However, the default data type for built-in generators (such as `np.zeros()` or `np.full()`) is `float64` (you may have already noticed this in previous examples). To change this data type, we need to pass `dtype` as an argument.

```python
arr = np.zeros(5, dtype=int)
print(arr)
```

To change an array’s data type once it has been defined, we can use the `astype()` method.

```python
arr = np.zeros(5)         # [0. 0. 0. 0. 0.]
arr = arr.astype(int)
print(arr)
```

### Indexing & Slicing

NumPy arrays support multiple ways to access and manipulate elements, similar to Python lists:

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

### Views vs Copies
NumPy arrays can either share memory (views) or have independent memory (copies):

- Views do not allocate new memory, meaning changes to the view affect the original array:

```python
a = np.array([1, 2, 3, 4])
v = a[1:3]  # view
v[0] = 20
print(a)
```

- Copies are created with `.copy()` and changes to a copy do not affect the original array:

```python
a = np.array([1, 2, 3, 4])
v = a[1:3].copy()
v[0] = 20
print(a)
```

---

## Array Attributes

### Array Properties

- `.shape` returns a tuple representing the dimensions of the array.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.shape)
```

- `.ndim` returns the number of dimensions (axes) in the array.

```python
arr = np.array([[[1, 1],
                 [2, 2],
                 [3, 3]]])
print(arr.ndim)
```

- `.size` returns the total number of elements in the array.

```python
arr = np.array([[1, 2, 3], 
                [4, 5, 6]])
print(arr.size)
```

- `.dtype` returns the data type of the elements.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.dtype)
```

- `.itemsize` returns the size (in bytes) of each element.
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.itemsize)
```

### Reshaping Arrays

- `.reshape(new_shape)` returns a view when possible; may return a copy. The total number of elements must remain the same.

```python
arr = np.arange(6)                # [0 1 2 3 4 5]
reshaped = arr.reshape((2, 3))
print(reshaped)
```

- `.ravel()` and `.flatten()` both change an array to 1D (if possible). `.ravel()` returns a view when possible; may return a copy (e.g., for non-contiguous arrays), whereas `.flatten()` always returns a copy of the array in 1D.

```python
arr = np.array([[1, 2],
                [3, 4]])

ravel = arr.ravel()               #[1 2 3 4]
ravel[0] = 99
print(arr)                        # original array is changed if ravel is a view

arr = np.array([[1, 2], 
                [3, 4]])
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

- `np.expand_dims(arr, axis)` adds a new axis at a specified position, increasing the number of dimensions by 1.

```python
arr = np.array([1, 2, 3])
expanded0 = np.expand_dims(arr, axis=0)  # adds axis at position 0
print(expanded0)

expanded1 = np.expand_dims(arr, axis=1)  # adds axis at position 1
print(expanded1)
```

### Stacking Arrays
Stacking arrays is the process of combining them along different axes:

- `np.hstack([a, b])` (or `np.concatenate([a, b], axis=1)`) stacks arrays horizontally.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.hstack([a, b]))
```

- `np.vstack([a, b])` (or `np.concatenate([a, b], axis=0)`) stacks arrays vertically.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.vstack([a, b]))
```

### Splitting Arrays
Splitting arrays is the process of dividing them into multiple sub-arrays.

- `np.split(arr, sections)` splits the array into equal-sized or specified sections (must divide evenly if integer).

```python
a = np.array([1, 2, 3, 4, 5, 6])

a1, a2, a3 = np.split(a, 3)
a4, a5, a6 = np.split(a, [3, 5])      # splits the array into arr[:3], arr[3:5], arr[5:]
print(a1, a2, a3)
print(a4, a5, a6)
```

- `np.array_split(arr, sections)` splits an array into unequal sections.

```python
a = np.array([1, 2, 3, 4, 5, 6])

a1, a2, a3, a4 = np.array_split(a, 4)
print(a1, a2, a3, a4)
```

---

## Basic Operations

NumPy arrays support **vectorized operations** meaning operations are applied **elementwise**, without the need for loops.

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

It can be used with boolean indexing, but this will flatten the array to 1D if you use a single mask of the same shape (e.g., arr[arr < 3]). However, if you use per-axis boolean masks (e.g., arr[row_mask, :] or arr[:, col_mask]), dimensions can be preserved.

```python
arr = np.array([[1, 5, 9],
                [4, 7, 2]])

arr2 = arr[arr < 3]
print(arr2)
```

### Aggregations
These are a group of methods that compute a single value over the whole array, or along a given axis.

- `.sum()` returns the sum of every element in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.sum())
print(arr.sum(axis=0))  # sum along columns (if 2D or higher)
print(arr.sum(axis=1))  # sum along rows (if 2D or higher)
```

- `.mean()` returns the mean of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.mean())
print(arr.mean(axis=0))
print(arr.mean(axis=1))
```

- `.max()` returns the maximum value in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.max())
print(arr.max(axis=0))
print(arr.max(axis=1))
```

- `.min()` returns the minimum value in the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.min())
print(arr.min(axis=0))
print(arr.min(axis=1))
```

- `.std()` returns the standard deviation of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.std())
print(arr.std(axis=0))
print(arr.std(axis=1))
```

- `.var()` returns the variance of the array

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr.var())
print(arr.var(axis=0))
print(arr.var(axis=1))
```

---

### Reductions with Axis and Keepdims

Many aggregation functions (sum, mean, min, max, std, var, etc.) support the `axis` parameter to compute results along a specific axis, and the `keepdims` parameter to retain the reduced dimension as size 1. This is often useful for broadcasting.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

col_means = arr.mean(axis=0)  # shape (3,)
row_means = arr.mean(axis=1)  # shape (2,)

# With keepdims=True, shapes are preserved for broadcasting
col_means_b = arr.mean(axis=0, keepdims=True)  # shape (1, 3)
row_means_b = arr.mean(axis=1, keepdims=True)  # shape (2, 1)

# Example: center columns by subtracting the mean of each column
centered = arr - arr.mean(axis=0, keepdims=True)
print(centered)
```

---

### Arg and Index Utilities

NumPy provides several functions to find the positions of maximum/minimum values, sort/order arrays, and extract uniqueness/counts:

- `np.argmax(arr, axis=None)` / `np.argmin(arr, axis=None)`: Index of the maximum/minimum value.
- `np.argsort(arr, axis=-1)`: Indices that would sort the array.
- `np.unique(arr, return_counts=False)`: Sorted unique values (optionally counts).
- `np.count_nonzero(arr, axis=None)`: Count of non-zero (or True) elements.
- `np.nonzero(arr)`: Indices where the array is non-zero (or True).
- `np.argwhere(arr)`: Indices where the condition is True (as an array).
- `np.bincount(arr)`: Count of occurrences of each integer value in an array of non-negative ints.

```python
arr = np.array([1, 3, 2, 3, 5, 1, 2])
print(np.argmax(arr))
print(np.argsort(arr))
print(np.unique(arr))
print(np.unique(arr, return_counts=True))
print(np.count_nonzero(arr > 2))
print(np.nonzero(arr == 3))
print(np.argwhere(arr > 2))
print(np.bincount(arr))
```

---

### NaN-aware Statistics

NumPy provides special versions of many statistical functions that ignore NaN (Not a Number) values.

- `np.nanmean(arr)`, `np.nanstd(arr)`, `np.nanmax(arr)`, `np.nanmin(arr)` ignore NaNs.
- `np.isnan(arr)` identifies NaNs in an array.
- `np.isfinite(arr)` identifies finite values (not NaN or ±inf).

```python
arr = np.array([1, np.nan, 3])
print(np.nanmean(arr))
print(np.isnan(arr))
print(np.isfinite(arr))
```

---

### Universal Function Parameters: out=, where=

Most universal functions (ufuncs, e.g., np.add, np.multiply) support the `out=` parameter (to write the result to a pre-allocated array) and `where=` (to mask where the operation applies):

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
out = np.empty_like(a)
np.add(a, b, out=out)            # result stored in 'out'
print(out)

# Only add where b > 4
result = np.add(a, b, where=b>4)
print(result)  # [1+4, 2+5, 3+6] but only where b>4; elsewhere original a is kept
```

---

## Random Numbers
NumPy’s `np.random` module provides tools for generating random numbers, sampling from distributions, and shuffling data.

### NumPy Random Distributions

- `np.random.rand(d0, d1, ...)` fills an array of the given dimensions with random floats in [0.0, 1.0).

```python
print(np.random.rand(2,2)) 
```

- `np.random.randn(d0, d1, ...)` fills an array of the given dimensions with random samples from a standard normal distribution (mean 0, std 1).

```python
print(np.random.randn(2,2,2)) 
```

- `np.random.randint(low, high, size)` fills an array of the given shape (`size`) with random integers between low (inclusive) and high (exclusive). 

```python
print(np.random.randint(1, 10, (2,2)))
```

- `np.random.seed(value)` sets a seed for the legacy random number generator so results are reproducible. Each integer will give a constant set of random numbers. For modern, safer workflows, use the Generator approach below.

```python
np.random.seed(0)
print(np.random.rand(2,2))
```

- `np.random.uniform(low, high, size)` fills an array of the given shape (`size`) with random samples from a given uniform distribution (between low and high).

```python
print(np.random.uniform(100, 120, (2,2)))
```

- `np.random.normal(loc, scale, size)` fills an array of the given shape (`size`) with random samples from a given normal (Gaussian) distribution (mean `loc`, standard deviation `scale`).

```python
print(np.random.normal(5, 2, (2,2)))
```

### Random Numbers using a Generator Object

The modern approach to generating random numbers is using a Generator object, which is recommended over the legacy `np.random.*`. This can also use a seed for reproducibility.

```python
rng = np.random.default_rng(seed=0)
print(rng.random(5))                        # 5 random floats in [0.0, 1.0)

arr = rng.random((2,2))                     # a 2x2 array filled with random floats in [0.0, 1.0)
```

The generator object supports a variety of distributions:

```python
# no-run
rng = np.random.default_rng()

rng.random()                          # uniform distribution in [0.0, 1.0)
rng.uniform(low, high)                 # uniform distribution between low and high
rng.integers(low, high)                # integers between low (inclusive) and high (exclusive)
rng.normal(loc=mean, scale=std)        # normal/Gaussian distribution with mean and standard deviation
rng.binomial(n, p)                     # binomial distribution with n trials and probability p
rng.poisson(lam)                       # Poisson distribution with expected value lambda
rng.exponential(scale=1/rate)          # exponential distribution with given rate (lambda = 1/scale)
rng.gamma(shape, scale)                # gamma distribution with shape (k) and scale (theta)
rng.beta(a, b)                         # beta distribution with parameters a and b
rng.multivariate_normal(mean, cov)     # multivariate normal distribution with mean vector and covariance matrix
```

If we add a `size` parameter to any of these, we can create an array with these distributions:

```python
rng = np.random.default_rng()
ints = rng.integers(1, 10, (2, 2))
print(ints)
```

```python
rng = np.random.default_rng()
normal = rng.normal(0, 1, 10)
print(normal)
```

```python
rng = np.random.default_rng()
beta = rng.beta(2, 5, (3, 3))
print(beta)
```

---

### Modern RNG Extras

The Generator object also has convenient methods for sampling and shuffling:

- `rng.choice(a, size=None, replace=True, p=None)`: Randomly sample elements from array `a`, with or without replacement, and (optionally) with probabilities.
- `rng.permutation(a)`: Returns a new randomly permuted array, leaving original unchanged.
- `rng.shuffle(a)`: Shuffles array `a` in-place along the first axis (rows for 2D arrays).

```python
rng = np.random.default_rng(42)
arr = np.arange(10)

sample = rng.choice(arr, size=5, replace=False)  # 5 unique elements, no replacement
print(sample)

weighted_sample = rng.choice(arr, size=5, p=np.linspace(0, 1, 10)/np.linspace(0, 1, 10).sum())
print(weighted_sample)

shuffled = rng.permutation(arr)
print(shuffled)

arr2 = np.arange(10)
rng.shuffle(arr2)
print(arr2)
```

---

### Shuffling

- `np.random.shuffle(arr)` shuffles an array **in place**. It has no return value and directly affects the original array. For modern code, prefer `rng.shuffle(arr)` as shown above.

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

Note: Shuffling a 2D array shuffles rows by default. To shuffle columns, use the transpose or modern Generator methods.

---

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

If you want to align two arrays so they can be broadcast, we can use `None` or `np.newaxis` to add a new axis.

```python
arr = np.array([1, 2, 3])  # shape (3,)

col_vec = arr[:, None]     # shape (3, 1)

row_vec = np.array([10, 20, 30])  # shape (3,)
result = col_vec + row_vec         # shape (3, 3)
print(result)
```

Reductions with `keepdims=True` can be especially helpful for broadcasting. For instance, see [Reductions with Axis and Keepdims](#reductions-with-axis-and-keepdims).

---

## Advanced Indexing

NumPy also supports more advanced indexing methods, such as boolean indexing and fancy indexing, which would require more verbose approaches like list comprehensions with native Python lists.

### Boolean Indexing
Selects elements from an array based on a condition, returning a 1D array of the elements that satisfy it (when using a mask of the same shape). Per-axis boolean masks (e.g., arr[row_mask, :]) can preserve dimensions.

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

print("Euclidean/L2 norm: ", np.linalg.norm(v))                              # = sqrt(3^2 + (-4)^2)
print("Manhattan/L1 norm: " , np.linalg.norm(v, 1))                          # = |3| + |-4|
print("Infinity norm (max absolute value): ", np.linalg.norm(v, np.inf) )    # = max(|3|, |-4|)
print("Frobenius norm/L2 norm for matrices: ", np.linalg.norm(A))            # = sqrt(2^2 + 4^2 + 5^2 + 1^2)
```

- **SVD (Singular Value Decomposition)**:  
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

## Statistics with NumPy

### Single Variable Statistics

NumPy provides various tools for common statistical operations.  
You have already seen how for an `np.array`, we can call methods such as `mean()` or `sum()` in [Aggregations](#aggregations). These also exist as functions:

```python
arr = np.array([1, 2, 3, 4, 5, 6])

print("Sum: ", np.sum(arr))
print("Mean:", np.mean(arr))
print("Maximum ", np.max(arr))
print("Minimum: ", np.min(arr))
print("Standard Deviation: ", np.std(arr))
print("Variance: ", np.var(arr))
```

We can also find further statistical data, not available as methods:

- `np.median(arr)` returns the median value in the array.

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(np.median(arr))
```

- `np.quantile(arr, q)` returns the value below which the fraction `q` of the data lies, interpolating if necessary.
- `np.percentile(arr, p)` returns the value below which the percentage `p` of the data lies, interpolating if necessary.

```python
arr = np.array([1, 2, 3, 4, 5, 6])
print(np.quantile(arr, 0.1))                # 10% of the data lies below this value
print(np.percentile(arr, 50))              # 50% of the data lies below this value = median
```

### Multivariate Statistics

- `np.corrcoef(x,y)` will return the correlation coefficient matrix, which measures linear relationships between variables (`-1` meaning perfectly negatively correlated to `1` perfectly positively correlated)

```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, -2, 1])
print(np.corrcoef(x, y))
print("r =", np.corrcoef(x, y)[0, 1])       # if the data is two 1D arrays, this will return the single value for the correlation coefficient
```

*Note that the correlation coefficient matrix is of the form $\begin{bmatrix} 1 && r \\ r && 1 \end{bmatrix}$ so either `[0, 1]` or `[1, 0]` gets the single value.*

- `np.cov(x,y)` will return the covariance matrix of two arrays, which measures how two variables vary together (positive meaning they increase/decrease together, negative meaning while one increases the other decreases)

```python
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, -2, 1])
print(np.cov(x, y))
print("cov(x, y) = ", np.cov(x, y)[0, 1])
```

*Note that the covariance matrix is of the form $\begin{bmatrix} var(x) && cov(x,y) \\ cov(y,x) && var(y) \end{bmatrix}$ so either `[0, 1]` or `[1, 0]` gets the single value.*

*Note also that the variance here is the sample variance (dividing by n-1) not the population variance (dividing by n, calculated by `x.var()` or `np.var(x)`). To get the population variance, use `np.cov(x, y, bias=True)`.*

---

## Saving and Loading Data
There are multiple ways to save and load data in NumPy, each with their own strengths and weaknesses.

- Saving and loading data using `.npy`:  
NumPy has its own binary format. It is very fast and recommended if you are only using NumPy, however often incompatible with other things.

```python
# no-run
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.save("arr.npy", arr)

...

loaded_arr = np.load("arr.npy")
```

- Saving and loading multiple arrays using `.npz`:  
Another native format is `.npz` which is a zipped collection of `.npy` files, useful for storing multiple arrays.

```python
# no-run
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])

arr2 = np.array([[100, 1000],
                 [200, 2000]])

np.savez("arr.npz", array1=arr1, array2=arr2)
np.savez_compressed("arr_compressed.npz", array1=arr1, array2=arr2)   # same as np.savez() but with compression, smaller file size but slower to load/save

...

loaded_data = np.load("arr.npz")

arr1 = loaded_data["array1"]
```

- Saving and Loading Text `.txt` or `.csv`:  
These file types are more compatible with other software (e.g. Microsoft Excel) but lose some NumPy-specific metadata.

```python
# no-run
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

np.savetxt(
    "data.csv", 
    arr, 
    delimiter=",", 
    fmt="%d", 
    header="Col1,Col2,Col3", 
    comments=""  # removes the default '#' before the header
)
```

This results in the file `data.csv` as:

```csv
Col1,Col2,Col3
1,2,3
4,5,6
```

Loading the file back into a NumPy array:

```python
# no-run
loaded_arr = np.loadtxt("data.csv", delimiter=",", skiprows=1)   # skiprows=1 ignores the header line
```

- `np.genfromtxt()` is similar to `np.loadtxt` but can handle missing data and more complex formats.
- For very large arrays, consider using `np.memmap` to memory-map array data from disk without loading it all into RAM.

---

## Polynomials
Moving beyond arrays, NumPy also has tools for working with polynomials.

- `np.poly1d(coefficients)` creates a polynomial object from a list of coefficients (in descending order of powers). These objects can be evaluated, added, multiplied or differentiated easily.

```python
p = np.poly1d([2, 3, 4])    # 2x^2 + 3x + 4
print(p(2))                 # evaluate at x=2
print(p.deriv())

q = p*2                     # q = 4x^2 + 6x + 8
```

- `np.polyfit(x, y, deg)` finds the least-squares polynomial of degree `deg` that fits the data `(x, y)`, returning the coefficients in descending powers.

```python
x = np.array([0, 1, 2, 3])
y = np.array([1, 3, 7, 13])

coeffs = np.polyfit(x, y, 2)   # finds a quadratic (degree 2) polynomial that best fits this data
print(coeffs)                  # x^2 + x + 1 fits this data best
```

- `np.polyval(p, x)` evaluates a polynomial `p` (as either a `poly1d` object or a list of coefficients) at given values of `x`.

```python
coeffs = [2, 3, 4]  # 2x^2 + 3x + 4
x = np.array([0, 1, 2, 3])

y = np.polyval(coeffs, x)
print(y)
```

---