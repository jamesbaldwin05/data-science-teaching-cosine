# Linear Algebra

Linear algebra covers vectors, matrices, and linear transformations. It's essential to data science because it's fundamental for handling datasets, performing computations efficiently, and powering techniques such as linear regression and neural networks.

---

## Table of Contents
1. [Vectors](#vectors)
    - [Vector Basics](#vector-basics)
    - [Basic Vector Operations](#basic-vector-operations)
    - [More About Vectors](#more-about-vectors)
2. [Matrices](#matrices)
    - [Matrices Basics](#matrices-basics)
    - [Basic Matrix Operations](#basic-matrix-operations)
    - [Matrix Multiplication](#matrix-multiplication)
    - [Special Matrices](#special-matrices)
    - [Determinant & Inverse](#determinant--inverse)
3. [Vector Spaces]()
4. [Linear Systems]()
5. [Eigenvalues and Eigenvectors](#eigenvalues--eigenvectors)

---

## Vectors

### Vector Basics

- A single number (like 4 or -2.834) is called a **scalar**.
- A **vector** is an ordered list of numbers (e.g. $\vec{a} = [2, 1, 8]$) and is used to represent points or directions in space.

**Coordinate Notation:**
- A vector in 3D: $\vec{v} = [v_1, v_2, v_3]$
- This is comparable to a coordinate point: $(x, y, z)$

```python
import numpy as np

v = np.array([2, 1, 8])
print("Vector v:", v)
```

---

### Basic Vector Operations

- **Addition/Subtraction**: Add/subtract corresponding elements of the vectors.  
  If $\vec{a} = [a_1, a_2, a_3]$ and $\vec{b} = [b_1, b_2, b_3]$, then:  
  $\vec{a} + \vec{b} = [a_1 + b_1, \; a_2 + b_2, \; a_3 + b_3]$

- **Scalar Multiplication**: Multiply each element by a scalar.  
  If c is a number and $\vec{a} = [a_1, a_2, a_3]$, then:  
  c $\vec{a} = [c a_1, \; c a_2, \; c a_3]$

- **Dot Product**: Measures how similar two vectors are (used in projections, similarity and machine learning). Result is a scalar, not a vector.  
  For $\vec{a} = [a_1, a_2, a_3]$ and $\vec{b} = [b_1, b_2, b_3]$:  
  $\vec{a}\cdot\vec{b} = a_1 b_1 + a_2 b_2 + a_3 b_3$

```python
import numpy as np

a = np.array([1, 2, 6])
b = np.array([3, 7, 2])

# Vector addition
print("a + b =", a + b)        # [1+3, 2+7, 6+2]

# Scalar multiplication
print("2 * a =", 2 * a)        # [2*1, 2*2, 2*6]

# Dot product
dot = np.dot(a, b)             # 1*3 + 2*7 + 6*2
print("Dot product:", dot)

```

---

### More About Vectors

- **Vector Norm (Length)**: Measures how long the vector is.
For $\vec{a} = [a_1, a_2, a_3]$, the norm $\left\|\vec{a}\right\| = \sqrt{{a_1}^2 + {a_2}^2 + {a_3}^2}$

```python
import numpy as np

a = np.array([1, 2, 6])
print("Norm: ", np.linalg.norm(a))
```

- **Unit vectors**: By dividing a vector by its norm (length), we can scale it to have a length of 1
For $\vec{a} = [a_1, a_2, a_3]$, the unit vector is given by $\hat{a} = \dfrac{\vec{a}}{\left\|\vec{a}\right\|}$.

```python
import numpy as np

a = np.array([1, 2, 6])
a_norm = np.linalg.norm(a)
a_unit = a / a_norm
print("Unit vector of a: ", a_unit)                # The norm of this vector is 1
```

- **Cosine Similarity**: Using the formula below, we can derive a relationship between the dot product and the angle between two vectors.  
For $\vec{a} = [a_1, a_2, a_3],\;\vec{b} = [b_1, b_2, b_3]$ and $\theta$ being the angle between $\vec{a}$ and $\vec{b}$  
$$
\vec{a}\cdot\vec{b} = \left\|\vec{a}\right\|\left\|\smash{\vec{b}}\right\|\cos{\theta}
$$

This can be rearranged to solve for the angle between two vectors:  
$$
\theta = \arccos{\left(\dfrac{\vec{a}\cdot\vec{b}}{\left\|\vec{a}\right\|\left\|\smash{\vec{b}}\right\|}\right)}
$$

```python
import numpy as np

a = np.array([1, 2, 6])
b = np.array([3, 7, 2])

a_norm = np.linalg.norm(a)
b_norm = np.linalg.norm(b)
dot_product = np.dot(a, b)

theta = np.arccos(dot_product/(a_norm * b_norm))
print("Angle between a and b (radians): ", theta)
```

*Radians are another way of measuring angles and there are $2\pi$ radians in $360^\circ$. The above angle is about $54.9^\circ$.*

- **Linear Independence**: A set of vectors {$\vec{v_1}, \vec{v_2}, ...., \vec{v_n}$} is linearly independent if no vector in the set can be written as a linear combination of the others.  
Formally, this is if the equation below only holds when all scalars $c_1,c_2,...,c_n=0$.
$$
c_1\vec{v_1} + c_2\vec{v_2} + ... + c_n\vec{v_n}= \vec{0}
$$

*The zero vector is written as $\vec{0}$ and is a n-th dimensional vector with every element in it 0*

- **Orthogonal Sets**: If the dot product of two vectors is 0, they are said to be orthogonal. Geometrically this can be thought of as the two vectors being perpendicular, since $\arccos(0)=\dfrac{\pi}{2}$.  
A set of vectors {$\vec{v_1}, \vec{v_2}, ...., \vec{v_n}$} is called an orthogonal set if every single pair of vectors in the set are orthogonal to each other. This means every vector in the set points in independent directions and orthogonal sets are always linearly independent.

- **Orthonormal Sets**: If each vector in an orthogonal set has unit length, the set is called an orthonormal set. These are useful for simplifying and speeding up computation. There are algorithms such as the Gram-Schmidt procedure that form an orthonormal set from a set of linearly independent vectors.

- **Projection**: The projection of a vector onto another vector tells us how much of the first vector lies in the direction of the second.  
For $\vec{a} = [a_1, a_2, a_3]$ and $\vec{b} = [b_1, b_2, b_3]$, the projection of $\vec{a}$ onto $\vec{b}$ is calculated by the formula:  
$$
\mathrm{proj}_{\vec{b}}\vec{a} = \left( \frac{\vec{a} \cdot \vec{b}}{\left\|\smash{\vec{b}}\right\|^2} \right) \vec{b}
$$

```python
import numpy as np

a = np.array([1, 2, 6])
b = np.array([3, 7, 2])

dot_product = np.dot(a, b)
b_norm_squared = np.dot(b, b)

projection = (dot_product / b_norm_squared) * b
print("Projection of a onto b:", projection)
```

- **Element-wise Operations**: This group of operations apply a function independently to each element of a vector. An example is the Hadamard product (element wise multiplication):  
For $\vec{a} = [a_1, a_2, a_3]$ and $\vec{b} = [b_1, b_2, b_3]$, the Hadamard product is $\vec{a} \ast \vec{b} = [a_1 b_1, \; a_2 b_2, \; a_3 b_3]$

```python
import numpy as np

a = np.array([1, 2, 6])
b = np.array([3, 7, 2])

elementwise_product = a * b
print("Element-wise product:", elementwise_product)
```

*In linear algebra theory, element-wise operations are often emphasized less since they are not linear transformations (they cannot be represented as a matrix) and do not preserve vector space structure. They do have practical applications to data science, especially under the hood.*

---

## Matrices

### Matrices Basics

- **Matrix**: A **matrix** is a 2D grid of numbers, written as rows and columns. A matrix with $m$ rows and $n$ columns is said to have shape $m\times n$. A vector is just a special case of a matrix (one with only one row or column).  
$A_{m\times n} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print("Shape:", A.shape)
```

---

### Basic Matrix Operations

- **Addition/Subtraction**: Add/subtract corresponding elements of the matrices. The matrices **must** be the same shape, you can add two $3 \times 3$ matrices but you cannot add a $3 \times 2$ matrix to a $4 \times 4$ matrix.  

```python
import numpy as np
A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

B = np.array([[7, 1, 2],
              [3, 6, 8],
              [5, 5, 1]])

print("A-B:")
print(A-B)
```

- **Scalar Multiplication**: Multiply each element by a scalar.

If $c$ is a scalar and $\vec{a} = [a_1, a_2, a_3]$, then $c\vec{a} = [c a_1, c a_2, c a_3]$.

```python
import numpy as np
A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

print("3A:")
print(3*A)
```

- **Transpose**: Swap the rows and columns of the matrix. This is denoted by $A^T$.

```python
import numpy as np
A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

print("Transpose of A:")
print(A.T)
```

---

### Matrix Multiplication

- Matrix multiplication works differently to other operations.
- Firstly, the number of columns of the first matrix **must** be the same as the number of rows of the second matrix.
- The resulting matrix has the number of rows of the first matrix and number of columns of the second matrix.
- Multiplying a matrix $A_{m\times p} \times B_{p\times n}$ results in a matrix of shape $m\times n$.

```python
import numpy as np

A = np.random.random((3, 2))      # Creates a 3x2 matrix filled with random numbers between 0 and 1
B = np.random.random((2, 1))

C = A @ B                         # @ is used for matrix multiplication although np.dot(A, B) or np.matmul(A, B) also work

print("Shape of result: ", C.shape)

```

- Secondly, matrix multiplication is not commutative meaning $AB \neq BA$ in general, even when both $A$ and $B$ are defined.
- It may not even be defined in certain cases: $A_{2\times 1} \times B_{1\times 4}$ is defined but $B_{1\times 4} \times A_{2\times 1}$ is not.
- There are cases where $AB = BA$ but these are rare and in general $AB$ and $BA$ will look very different.

```python
import numpy as np

A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

B = np.array([[7, 1, 2],
              [3, 6, 8],
              [5, 5, 1]])

print("AB:")
print(A @ B)
print("BA:")
print(B @ A)
```

- The algorithm for multiplying two matrices involves a lot of calculations and is therefore quite long to explain (although the calculations are simple and it will not take long to understand).
- It is probably worth learning if you are new to matrices entirely. There is a video [here](https://www.youtube.com/watch?v=2spTnAiQg4M) with a clear explanation.
It will be ignored here since numpy can do all these calculations very fast and under the hood anyway.

- To multiply a matrix by a vector, use the algorithm below:  
For a matrix $A_{m\times n} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$ and a vector $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$,  
$A\vec{v} = \begin{bmatrix}
a_{11}v_1 + a_{12}v_2 + \cdots + a_{1n}v_n \\
a_{21}v_1 + a_{22}v_2 + \cdots + a_{2n}v_n \\
\vdots \\
a_{m1}v_1 + a_{m2}v_2 + a_{mn}v_n
\end{bmatrix}$

- Note that the result is a vector.
- This is equivalent to doing the dot product of each row in the matrix with the vector.

```python
import numpy as np

A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

v = np.array([1, 2, 3])

print("Multiplying A and v: ")
print(A @ v)

```

---

### Special Matrices

- **Zero matrix**: A matrix where all elements are zero.  
For any matrix $A$, $\;A + 0 = A$ and $A \times 0 = 0$ (assuming the dimensions are compatible).
- **Square matrix**: A matrix with the same number of rows and columns (e.g. a $3\times 3$ matrix).
- **Identity matrix**: A square matrix with 1s on the diagonal (top left to bottom right) and 0s elsewhere.  
For any matrix $A$, $\; AI = IA = A$ (assuming the dimensions are compatible).  
For example, the $3\times 3$ identity matrix is:  
$I_3 = \begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1\end{bmatrix}$
- **Diagonal matrix**: A square matrix with all off-diagonal elements 0.   
For example:
$A = \begin{bmatrix}
-4 & 0 & 0\\
0 & 3 & 0\\
0 & 0 & 2\end{bmatrix}$
- **Scalar matrix**: A diagonal matrix with all diagonal elements the same scalar and all off-diagonal elements 0. Can be written as multiples of the identity matrix.  
For example:
$A = \begin{bmatrix}
5 & 0 & 0\\
0 & 5 & 0\\
0 & 0 & 5\end{bmatrix} = 5I$
- **Upper Triangular matrix**: A square matrix where all the elements below the main diagonal are 0.     
For example:
$A = \begin{bmatrix}
-3 & -3 & 1\\
0 & 9 & -7\\
0 & 0 & 5\end{bmatrix}$
- **Lower Triangular matrix**: A square matrix where all the elements above the main diagonal are 0.     
For example:
$A = \begin{bmatrix}
-3 & 0 & 0\\
9 & -2 & 0\\
4 & 3 & 5\end{bmatrix}$
- **Symmetric matrix**: A square matrix that is equal to its transpose ($A=A^T$).  
For example,
$A = \begin{bmatrix}
2 & 3 & 4\\
3 & 5 & 6\\
4 & 6 & 1\end{bmatrix}$
- **Skew-symmetric matrix**: A square matrix that is equal to its negative transpose ($-A=A^T$). The diagonal is always 0.  
For example,
$A = \begin{bmatrix}
0 & 2 & -4\\
-2 & 0 & 6\\
4 & -6 & 0\end{bmatrix}$
- **Orthogonal matrix**: A square matrix whose transpose is also its inverse ($A^T=A^{-1}$). Columns (and rows) are orthonormal vectors.
- **Singular matrix**: A square matrix that does not have an inverse (its determinant is 0).
- **Diagonalizable matrix**: A matrix that can be written as $A = P D P^{-1}$ where $D$ is a diagonal matrix.  
*More on inverses, determinants and diagonalizable matrices soon.*

```python
import numpy as np
A = np.zeros((2, 3))    # 2x3 zero matrix
I = np.eye(3)           # 3x3 identity
D = np.diag([1, 2, 3])  # Diagonal matrix
print(D)
```

---

### Determinant & Inverse

- **Determinant**: A single number summarizing a square matrix. If $\det(A) = 0$, matrix can't be inverted.

```python
import numpy as np

A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

print("Determinant of A:", np.linalg.det(A))
```

- **Inverse**: $A^{-1}$ "undoes" $A$ (if it exists) i.e. $A A^{-1} = I$. Used for solving systems of equations.

```python
import numpy as np

A = np.array([[3, 5, 2],
              [1, 8, 3],
              [4, 4, 7]])

print("Inverse of A:")
print(np.linalg.inv(A))
```

---

## Eigenvalues and Eigenvectors

- As discussed, matrices are essentially just a way of representing a linear transformation.  
For $A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} x \\ y \end{bmatrix}$, $\;A\vec{v} = \begin{bmatrix} 4x+y \\ 2x+3y \end{bmatrix}$

- This means if we can find a way of representing the same linear transformation with a different matrix, one that simply scales vectors along special directions, then understanding and processing the matrix becomes a lot easier.

- These special directions are called eigenvectors and the scaling factors are called eigenvalues.  In other words, for eigenvector $\vec{v}$ and eigenvalue $\lambda$, $\; A\vec{v}= \lambda \vec{v}$, which means that the transformation $A$ simply stretches the eigenvector without changing its direction. Each eigenvector corresponds to an eigenvalue and vice versa.

- Eigenvectors and eigenvalues exist for every square matrix if we allow for complex numbers, but they are not always real.   

- However, they can be used to represent a matrix in diagonal form which, due to the large number of zeros, greatly speeds up computation time.

- For example, consider the matrix $A = \begin{bmatrix} 4 & 1\\2&3 \end{bmatrix}$. The eigenvalues for this matrix are $5$ and $2$ (these can be calculated on a computer or using methods such as the characteristic equation). To find the eigenvectors for this matrix, we solve for each eigenvalue individually:  
For $\lambda = 5$,  
$\begin{bmatrix} 4&1\\2&3 \end{bmatrix} \vec{v} = 5\; \vec{v}$  
$\begin{bmatrix} 4&1\\2&3 \end{bmatrix} \vec{v} - 5\; I\;\vec{v} = \vec{0}$  
$(\begin{bmatrix} 4&1\\2&3 \end{bmatrix} - 5\; \begin{bmatrix} 1&0\\0&1 \end{bmatrix})\vec{v} = \vec{0}$  
$\begin{bmatrix} -1&1\\2&-2 \end{bmatrix} \begin{bmatrix} x\\ y \end{bmatrix} = \begin{bmatrix} 0\\ 0 \end{bmatrix}$  
$-x+y=0,\; 2x-2y=0$ solved simultaneously gives $x=1,\; y=1$  
So for eigenvalue $\lambda=5$, the eigenvector is $\begin{bmatrix} 1\\ 1 \end{bmatrix}$.  
For $\lambda=2$, the eigenvector is $\begin{bmatrix} 1\\ -2 \end{bmatrix}$.  

- We can now rewrite $A$ as a diagonal matrix using the formulas $A = P D P^{-1}$ and $D = P^{-1} A P$, where $P$ is the matrix formed by using the eigenvectors as columns of the matrix (and $P^{-1}$ is the inverse of this matrix). In our example,  
$P = \begin{bmatrix} 1&1\\1&-2 \end{bmatrix}$ and  $P^{-1} = \begin{bmatrix} 2/3&1/3\\1/3&-1/3 \end{bmatrix}$.

- $D$ is the matrix with the eigenvalues corresponding to $P$ across the diagonal, in our example,  
$D = \begin{bmatrix} 5&0\\0&2 \end{bmatrix}$.

- Using all this, we can show that in our example, $A = P D P^{-1}$ and $D = P^{-1} A P$.

- This is powerful in speeding up various computational processes as diagonal matrices are much easier to compute with than other types of matrix.

---

### You should be able to:
- Explain scalars, vectors, matrices, and their shapes.
- Compute vector and matrix operations in Python.
- Understand matrix multiplication and broadcasting.
- Recognize the role of eigenvalues/eigenvectors (e.g. in PCA).
- Interpret basic matrix concepts (identity, inverse).
---