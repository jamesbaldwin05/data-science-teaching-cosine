# Essential Mathematics for Data Science

## Why learn math for data science?
- Math underpins how we represent data, build models and draw conclusions
- No prior university-level maths required. We'll build intuition and show practical code for all key concepts.

---

## Table of Contents
1. [Linear Algebra](#linear-algebra)
   - [Vectors](#vectors)
   - [Matrices](#matrices)
   - [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
   - [PCA Example](#pca-example-principal-component-analysis-dimensionality-reduction)
2. [Calculus for Machine Learning](#calculus-for-ml-lightweight)
3. [Probability](#probability)
4. [Statistics](#statistics)
5. [Mathematical Notation Reference](#mathematical-notation-reference)
6. [Key Takeaways](#key-takeaways)
7. [Exercises & Mini-Projects](#exercises--mini-projects)

---

## Linear Algebra

Linear algebra covers vectors, matrices, and linear transformations. It's essential to data science because it's fundamental for handling datasets, performing computations efficiently, and powering techniques such as linear regression and neural networks.

---

### Vectors

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

#### Basic Vector Operations

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

#### More About Vectors

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

### Matrices

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

#### Basic Matrix Operations

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

#### Matrix Multiplication

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

#### Special Matrices

- **Zero matrix**: A matrix where all elements are zero.  
For any matrix $A$, $\;A + 0 = A$ and $A \times 0 = 0$ (assuming the dimensions are compatible).
- **Square matrix**: A matrix with the same number of rows and columns (e.g. a $3\times 3$ matrix).
- **Identity matrix**: A square matrix with 1s on the diagonal (top left to bottom right) and 0s elsewhere. For any matrix $A$, $\; AI = IA = A$ (assuming the dimensions are compatible).  
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

#### Broadcasting
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

#### Determinant & Inverse

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

### Eigenvalues & Eigenvectors

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

### PCA Example: Principal Component Analysis (Dimensionality Reduction)
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

#### You should be able to:
- Explain scalars, vectors, matrices, and their shapes.
- Compute vector and matrix operations in Python.
- Understand matrix multiplication and broadcasting.
- Recognize the role of eigenvalues/eigenvectors (e.g. in PCA).
- Interpret basic matrix concepts (identity, inverse).

---

## Calculus for ML (Lightweight)

**What:** The math of change—used to optimize, minimize error, and train models.

**Why:** Powers gradient descent and learning in ML.

### Derivative Concept & Slope Intuition

- **Derivative**: Rate of change; slope of a function at a point.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = x ** 2
plt.plot(x, y, label="y = x^2")
plt.plot(1, 1, 'ro', label="Point (1,1)")
plt.arrow(1, 1, 1, 2, head_width=0.1, head_length=0.2, color='red')
plt.title("Slope at x=1 is 2")
plt.legend()
plt.show()
```

---

### Gradients and Partial Derivatives

- **Gradient**: Vector of partial derivatives; points in direction of greatest increase.
- **Partial derivative**: Rate of change with respect to one variable, keeping others fixed.

```python
def f(x, y):
    return x**2 + y**2

# Gradient at (1,1):
df_dx = 2 * 1
df_dy = 2 * 1
print("Gradient at (1,1):", (df_dx, df_dy))
```

---

### Gradient Descent Walkthrough

- **What?** Iterative method to find minimum of a function.
- **Why?** Used to train ML models by minimizing loss.

```python
import matplotlib.pyplot as plt

# Minimize f(x) = x^2 + 2x + 1
f = lambda x: x**2 + 2*x + 1
df = lambda x: 2*x + 2

x = 5.0
learning_rate = 0.1
xs, ys = [], []

for i in range(20):
    xs.append(x)
    ys.append(f(x))
    x -= learning_rate * df(x)

plt.plot(xs, ys, marker='o')
plt.xlabel("Step")
plt.ylabel("f(x)")
plt.title("Gradient Descent Progress")
plt.show()
```

---

*Link to ML: Training most models = minimize a loss function using gradient descent or its variants.*

---

## Probability

**What:** The math of uncertainty—quantifying how likely events are.

**Why:** Essential for modeling randomness, making predictions, and drawing conclusions from incomplete information.

### Basic Probability Rules

- **Addition Rule**: $P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$
- **Multiplication Rule**: $P(A \text{ and } B) = P(A) \times P(B|A)$

**Example:**
If $P(A) = 0.2$, $P(B) = 0.5$, $P(A \text{ and } B) = 0.1$:
- $P(A \text{ or } B) = 0.2 + 0.5 - 0.1 = 0.6$

---

### Conditional Probability & Bayes' Theorem

- **Conditional**: Probability of $A$ given $B$: $P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$
- **Bayes' theorem**: Updates beliefs: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

**Concrete Example: Disease Testing**

Suppose:
- 1% of people have disease ($P(D) = 0.01$)
- Test is 99% accurate (true positive rate $P(+|D) = 0.99$; false positive $P(+|\neg D)=0.01$)
- You test positive. What is $P(D|+)$?

```python
# Bayes theorem calculation
P_D = 0.01      # Disease prevalence
P_pos_D = 0.99  # True positive rate
P_pos_notD = 0.01 # False positive rate

P_notD = 1 - P_D
P_pos = P_pos_D * P_D + P_pos_notD * P_notD
P_D_pos = (P_pos_D * P_D) / P_pos
print("Probability actually sick if test is positive:", P_D_pos)
```

---

### Independence

- **Events A and B are independent** if $P(A|B) = P(A)$.
- **Intuition**: If knowing B tells you nothing about A, they are independent.

---

### Key Discrete Distributions

- **Bernoulli**: One trial, yes/no (coin flip).
- **Binomial**: Repeated Bernoulli trials (e.g., number of heads in 10 flips).
- **Poisson**: Number of rare events in fixed time/space (e.g., calls per minute).

```python
# Bernoulli, Binomial, Poisson distributions
from scipy.stats import bernoulli, binom, poisson
import matplotlib.pyplot as plt

outcomes = bernoulli.rvs(0.7, size=10)
print("Bernoulli samples:", outcomes)

# Binomial: # of successes in 10 trials
binom_sample = binom.rvs(n=10, p=0.5, size=1000)
plt.hist(binom_sample, bins=11, alpha=0.7)
plt.title("Binomial Sample Distribution")
plt.show()

# Poisson: e.g., number of emails per hour
poisson_sample = poisson.rvs(mu=3, size=1000)
plt.hist(poisson_sample, bins=range(10), alpha=0.7)
plt.title("Poisson Sample Distribution")
plt.show()
```

---

### Key Continuous Distributions

- **Uniform**: All outcomes equally likely (e.g., random number between 0 and 1).
- **Normal**: Bell curve, real-world heights, test scores.
- **Exponential**: Time between random events (e.g., waiting time for bus).

```python
from scipy.stats import uniform, expon
import matplotlib.pyplot as plt

# Uniform
plt.hist(uniform.rvs(loc=0, scale=1, size=1000), bins=20, alpha=0.7)
plt.title("Uniform Distribution")
plt.show()

# Exponential
plt.hist(expon.rvs(scale=1, size=1000), bins=20, alpha=0.7)
plt.title("Exponential Distribution")
plt.show()
```

#### Why do these appear?
- Uniform: True randomness (e.g., random sampling).
- Normal: Sums/averages of many small effects (Central Limit Theorem).
- Exponential: Waiting for the next random event (memoryless property).

---

### Practical Tips for Choosing a Distribution

- **Count data?** Try Binomial (fixed n) or Poisson (unbounded).
- **Continuous, bell-shaped?** Try Normal.
- **Waiting times?** Try Exponential.
- **Simple yes/no?** Try Bernoulli.

---

## Calculus for ML (Lightweight)

**What:** The math of change—used to optimize, minimize error, and train models.

**Why:** Powers gradient descent and learning in ML.

### Derivative Concept & Slope Intuition

- **Derivative**: Rate of change; slope of a function at a point.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = x ** 2
plt.plot(x, y, label="y = x^2")
plt.plot(1, 1, 'ro', label="Point (1,1)")
plt.arrow(1, 1, 1, 2, head_width=0.1, head_length=0.2, color='red')
plt.title("Slope at x=1 is 2")
plt.legend()
plt.show()
```

---

### Gradients and Partial Derivatives

- **Gradient**: Vector of partial derivatives; points in direction of greatest increase.
- **Partial derivative**: Rate of change with respect to one variable, keeping others fixed.

```python
def f(x, y):
    return x**2 + y**2

# Gradient at (1,1):
df_dx = 2 * 1
df_dy = 2 * 1
print("Gradient at (1,1):", (df_dx, df_dy))
```

---

### Gradient Descent Walkthrough

- **What?** Iterative method to find minimum of a function.
- **Why?** Used to train ML models by minimizing loss.

```python
import matplotlib.pyplot as plt

# Minimize f(x) = x^2 + 2x + 1
f = lambda x: x**2 + 2*x + 1
df = lambda x: 2*x + 2

x = 5.0
learning_rate = 0.1
xs, ys = [], []

for i in range(20):
    xs.append(x)
    ys.append(f(x))
    x -= learning_rate * df(x)

plt.plot(xs, ys, marker='o')
plt.xlabel("Step")
plt.ylabel("f(x)")
plt.title("Gradient Descent Progress")
plt.show()
```

---

*Link to ML: Training most models = minimize a loss function using gradient descent or its variants.*

---

## Probability

**What:** The math of uncertainty—quantifying how likely events are.

**Why:** Essential for modeling randomness, making predictions, and drawing conclusions from incomplete information.

### Basic Probability Rules

- **Addition Rule**: $P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)$
- **Multiplication Rule**: $P(A \text{ and } B) = P(A) \times P(B|A)$

**Example:**
If $P(A) = 0.2$, $P(B) = 0.5$, $P(A \text{ and } B) = 0.1$:
- $P(A \text{ or } B) = 0.2 + 0.5 - 0.1 = 0.6$

---

### Conditional Probability & Bayes' Theorem

- **Conditional**: Probability of $A$ given $B$: $P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$
- **Bayes' theorem**: Updates beliefs: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

**Concrete Example: Disease Testing**

Suppose:
- 1% of people have disease ($P(D) = 0.01$)
- Test is 99% accurate (true positive rate $P(+|D) = 0.99$; false positive $P(+|\neg D)=0.01$)
- You test positive. What is $P(D|+)$?

```python
# Bayes theorem calculation
P_D = 0.01      # Disease prevalence
P_pos_D = 0.99  # True positive rate
P_pos_notD = 0.01 # False positive rate

P_notD = 1 - P_D
P_pos = P_pos_D * P_D + P_pos_notD * P_notD
P_D_pos = (P_pos_D * P_D) / P_pos
print("Probability actually sick if test is positive:", P_D_pos)
```

---

### Independence

- **Events A and B are independent** if $P(A|B) = P(A)$.
- **Intuition**: If knowing B tells you nothing about A, they are independent.

---

### Key Discrete Distributions

- **Bernoulli**: One trial, yes/no (coin flip).
- **Binomial**: Repeated Bernoulli trials (e.g., number of heads in 10 flips).
- **Poisson**: Number of rare events in fixed time/space (e.g., calls per minute).

```python
# Bernoulli, Binomial, Poisson distributions
from scipy.stats import bernoulli, binom, poisson
import matplotlib.pyplot as plt

outcomes = bernoulli.rvs(0.7, size=10)
print("Bernoulli samples:", outcomes)

# Binomial: # of successes in 10 trials
binom_sample = binom.rvs(n=10, p=0.5, size=1000)
plt.hist(binom_sample, bins=11, alpha=0.7)
plt.title("Binomial Sample Distribution")
plt.show()

# Poisson: e.g., number of emails per hour
poisson_sample = poisson.rvs(mu=3, size=1000)
plt.hist(poisson_sample, bins=range(10), alpha=0.7)
plt.title("Poisson Sample Distribution")
plt.show()
```

---

### Key Continuous Distributions

- **Uniform**: All outcomes equally likely (e.g., random number between 0 and 1).
- **Normal**: Bell curve, real-world heights, test scores.
- **Exponential**: Time between random events (e.g., waiting time for bus).

```python
from scipy.stats import uniform, expon
import matplotlib.pyplot as plt

# Uniform
plt.hist(uniform.rvs(loc=0, scale=1, size=1000), bins=20, alpha=0.7)
plt.title("Uniform Distribution")
plt.show()

# Exponential
plt.hist(expon.rvs(scale=1, size=1000), bins=20, alpha=0.7)
plt.title("Exponential Distribution")
plt.show()
```

#### Why do these appear?
- Uniform: True randomness (e.g., random sampling).
- Normal: Sums/averages of many small effects (Central Limit Theorem).
- Exponential: Waiting for the next random event (memoryless property).

---

### Practical Tips for Choosing a Distribution

- **Count data?** Try Binomial (fixed n) or Poisson (unbounded).
- **Continuous, bell-shaped?** Try Normal.
- **Waiting times?** Try Exponential.
- **Simple yes/no?** Try Bernoulli.

---

## Statistics

**What:** The study of data: summarizing, visualizing, and drawing conclusions.

**Why:** All data science starts with understanding data distributions, patterns, and variation.

### Descriptive Statistics

- **Mean**: Average.
- **Median**: Middle value.
- **Mode**: Most frequent value.
- **Variance**: How spread out data is.
- **Standard deviation (std)**: Typical distance from mean.
- **IQR (Interquartile Range)**: Range of the middle 50%.

```python
import pandas as pd

data = [5, 6, 7, 8, 8, 8, 10, 15]
s = pd.Series(data)
print("Mean:", s.mean())
print("Median:", s.median())
print("Mode:", s.mode().tolist())
print("Variance:", s.var())
print("Std:", s.std())
print("IQR:", s.quantile(0.75) - s.quantile(0.25))
```

---

### Visualizing Distributions

- **Histogram**: Shows frequency of values.
- **Boxplot**: Shows median, quartiles, outliers.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(data, kde=True)
plt.title("Histogram")
plt.show()

sns.boxplot(y=data)
plt.title("Boxplot")
plt.show()
```

---

### Probability Distributions Overview

- **Normal (Gaussian)**: Bell-shaped, many natural phenomena.
- **Binomial**: Counts successes in a series of yes/no trials.
- **Poisson**: Counts events in fixed time/space (rare events).

#### Plots

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, loc=0, scale=1), label="Normal")
plt.title("Normal Distribution (PDF)")
plt.legend()
plt.show()

k = np.arange(0, 11)
plt.stem(k, binom.pmf(k, n=10, p=0.5), use_line_collection=True)
plt.title("Binomial Distribution (PMF)")
plt.show()

lmbda = 3
k = np.arange(0, 10)
plt.stem(k, poisson.pmf(k, lmbda), use_line_collection=True)
plt.title("Poisson Distribution (PMF)")
plt.show()
```

---

### Sampling & Central Limit Theorem (CLT)

- **Sampling**: Drawing subsets from data.
- **CLT**: Means of samples from any distribution become bell-shaped (normal) as sample size grows.

#### Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
population = np.random.exponential(scale=2, size=10000)
means = [np.mean(np.random.choice(population, size=30)) for _ in range(1000)]

plt.hist(means, bins=30, color='orange', alpha=0.7)
plt.title("Sampling Distribution of Mean (CLT in action!)")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.show()
```

---

### Hypothesis Testing

- **What?** Test a claim about data (e.g., "mean is 0").
- **Z-test/T-test**: Compare means.
- **p-value**: Probability of seeing data as extreme as observed, assuming null hypothesis is true.

```python
import numpy as np
from scipy.stats import ttest_1samp

sample = np.random.normal(loc=1, scale=1, size=30)
t_stat, p_val = ttest_1samp(sample, popmean=0)
print("t-statistic:", t_stat)
print("p-value:", p_val)
```

*If p-value < 0.05, often considered "statistically significant" (but always interpret in context!).*

---

### Confidence Intervals

- **What?** Range likely to contain the true parameter (e.g., mean), with some confidence (e.g., 95%).
- **Python Example:**

```python
import numpy as np
import scipy.stats as stats

sample = np.random.normal(loc=1, scale=1, size=30)
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)
n = len(sample)
conf_int = stats.t.interval(0.95, n-1, loc=sample_mean, scale=sample_std/np.sqrt(n))
print("95% Confidence Interval:", conf_int)
```

---

### Correlation vs. Covariance

- **Covariance**: How two variables vary together (units matter).
- **Correlation**: Standardized covariance, ranges [-1, 1]. 1 = perfect positive, -1 = perfect negative.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})
print("Covariance:\n", df.cov())
print("Correlation:\n", df.corr())
```

#### Visualizing with a Heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()
```

---

#### You should be able to:
- Compute and interpret mean, median, mode, variance, std, IQR.
- Visualize data with histograms and boxplots.
- Identify and plot key probability distributions.
- Explain and simulate the Central Limit Theorem.
- Run a t-test and interpret p-values.
- Calculate and interpret correlation and covariance.

---

## Mathematical Notation Reference

| Symbol     | Name/Meaning                              | Plain English                | Python Equivalent            |
|------------|-------------------------------------------|------------------------------|------------------------------|
| $∑$        | Summation                                 | Add over index (e.g., $∑_i x_i$) | `sum(x)`                 |
| $∏$        | Product                                   | Multiply over index          | `np.prod(x)`                 |
| $∇$        | Nabla, gradient                           | Vector of partial derivatives| `np.gradient()`, manual      |
| $⊤$        | Transpose                                 | Flip rows/columns            | `.T` (NumPy)                 |
| $||·||$    | Norm                                      | Length/magnitude of vector   | `np.linalg.norm()`           |
| $𝔼[·]$     | Expectation                               | Average/mean                 | `np.mean()`                  |
| $Var(·)$   | Variance                                  | Spread of values             | `np.var()`                   |
| $Cov(·)$   | Covariance                                | How two variables vary together | `np.cov()`               |

---

## Key Takeaways

- **Linear algebra** is the language of data: vectors, matrices, transformations, and PCA.
- **Statistics** helps you summarize, visualize, and draw rigorous conclusions from data.
- **Probability** models uncertainty and randomness—crucial for inference and prediction.
- **Calculus** underpins optimization and learning in ML.
- **Notation** is a compact way to express big ideas—knowing how to read it empowers you.

---

## Exercises & Mini-Projects

Try these to practice your new math skills! (See the [Python lesson](01_python.md) for tips on using notebooks or scripts.)

1. **Vector & Matrix Gym**:  
   - Create two vectors and a matrix in NumPy.
   - Compute their sum, dot product, and multiply matrix × vector.
   - Change dimensions and explore what errors you get.

2. **Statistics Detective**:  
   - Download a small CSV (e.g., from the `data/` folder).
   - Compute mean, median, mode, std, and IQR for at least two columns.
   - Make a histogram and boxplot for both.

3. **Probability Simulator**:  
   - Simulate flipping a biased coin 1000 times.
   - Plot the running proportion of heads.
   - Compare to expected Binomial probability.

4. **Gradient Descent Animation**:  
   - Implement gradient descent to minimize $f(x) = (x-2)^2 + 3$ starting from $x=10$.
   - Plot the value of $x$ and $f(x)$ at each step.

5. **Mini PCA on Real Data**:  
   - Load the Iris dataset from `sklearn.datasets`.
   - Perform PCA, plot the explained variance (scree plot) and a scatterplot of the first two principal components.

*Want to go further?*  
- Try using SciPy's `curve_fit` to fit a curve to noisy data.
- Write a function that computes the variance and standard deviation *by hand* (no NumPy).

---

*Next steps:*  
- Keep practicing! Math is a skill—use it regularly and it will become second nature.
- Ready to move on? Check out the next lesson on R basics.

---