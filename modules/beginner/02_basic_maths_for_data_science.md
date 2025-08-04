# Essential Mathematics for Data Science

## Why learn math for data science?
- Math underpins how we represent data, build models and draw conclusions
- No prior university-level math required. We'll build intuition and show practical code for all key concepts.

---

## Table of Contents
1. [Linear Algebra](#linear-algebra)
2. [Statistics](#statistics)
3. [Probability](#probability)
4. [Calculus for Machine Learning](#calculus-for-ml-lightweight)
5. [Mathematical Notation Reference](#mathematical-notation-reference)
6. [Key Takeaways](#key-takeaways)
7. [Exercises & Mini-Projects](#exercises--mini-projects)

---

## Linear Algebra

Linear algebra covers vectors, matricies and linear transformations. It's essential to data science because it's fundamental for handling datasets, performing computations efficiently and powering techniques such as linear regression and neural networks.

### Vectors

- A single number (4, -2.834) is called a scalar
- A vector is an ordered list of numbers (e.g. $\mathbf[2, 1, 8]$) and is used to represent points and directions

**Coordinate Notation:**
- A vector in 3D: $\mathbf{v} = [v_1, v_2, v_3]$ 
- It is clearly comparable to a coordinate $\mathbf(x, y, z)$

```python
import numpy as np

v = np.array([2, 1, 8])
print("Vector v:", v)
```

### Vector Operations

- **Addition**: Add corresponding elements.
- **Scalar multiplication**: Multiply each element by a number.
- **Dot product**: Measures how much two vectors point in the same direction.

```python
a = np.array([1, 2])
b = np.array([3, 4])

# Addition
print("a + b =", a + b)

# Scalar multiplication
print("2 * a =", 2 * a)

# Dot product
dot = np.dot(a, b)
print("Dot product:", dot)
```

#### *What/Why*: Dot product tells you about similarity and projections—used in ML for similarity, attention, and projections.

---

### Matrices

- **Matrix**: A 2D grid of numbers (shape: rows × columns, e.g., 3 × 2).
- **Notation**: $A_{ij}$ is the entry in row $i$, column $j$.
- **Shape**: Use `.shape` in NumPy.

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print("Shape:", A.shape)  # (3, 2)
```

#### Matrix Transpose

- Flips rows and columns: $A^\top$ ("A transposed").

```python
A_T = A.T
print(A_T)
```

---

### Matrix Multiplication & Broadcasting

- **Matrix multiplication**: Combines two matrices (or matrix × vector).
- **What/Why**: Used to transform data, combine features, apply weights in neural nets.
- **Rule**: $(m \times n)$ × $(n \times p)$ → $(m \times p)$

```python
B = np.array([[1, 2, 3],
              [4, 5, 6]])
# A: (3,2), B: (2,3)
AB = np.dot(A, B)
print(AB)
```

#### Broadcasting

- **What?** NumPy automatically expands arrays to match shapes for element-wise ops.
- **Why?** Lets you write concise, efficient code.

```python
# Add a vector to each row of a matrix
A = np.array([[1, 2], [3, 4], [5, 6]])
v = np.array([10, 100])
print(A + v)
```

---

### Special Matrices

- **Identity matrix ($I$)**: Diagonal of 1s, rest 0s. Acts as "1" for matrices.
- **Diagonal matrix**: Only entries on the diagonal are non-zero.

```python
I = np.eye(3)           # 3x3 identity
D = np.diag([1, 2, 3])  # Diagonal matrix
print(I)
print(D)
```

---

### Determinant & Inverse

- **Determinant**: A single number summarizing a square matrix. If $|A| = 0$, matrix can't be inverted.
- **Inverse**: $A^{-1}$ "undoes" $A$ (if it exists). $A A^{-1} = I$
- **When do you need them?** Inverting matrices is used for solving equations, but in data science it's often avoided for speed/stability.

```python
from numpy.linalg import det, inv

M = np.array([[4, 7],
              [2, 6]])
print("Determinant:", det(M))
print("Inverse:\n", inv(M))
```

---

### Eigenvalues & Eigenvectors (+ PCA Demo)

- **What?** Eigenvectors are directions that stay the same when a matrix is applied; eigenvalues tell how much they're stretched.
- **Why?** Foundational for PCA (dimensionality reduction), stability, understanding transformations.

#### PCA Example: Principal Component Analysis (Dimensionality Reduction)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Toy dataset: 100 points in 2D
np.random.seed(0)
X = np.dot(np.random.rand(100, 2), np.array([[3, 1], [1, 2]])) + np.array([5, 10])

pca = PCA()
pca.fit(X)
explained = pca.explained_variance_ratio_

plt.figure(figsize=(5, 3))
plt.bar([1, 2], explained, color='skyblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.show()
```

*PCA finds directions ("principal components") that capture the most variance in data.*

---

#### You should be able to:
- Explain scalars, vectors, matrices, and their shapes.
- Compute vector and matrix operations in Python.
- Understand matrix multiplication and broadcasting.
- Recognize the role of eigenvalues/eigenvectors (e.g., in PCA).
- Interpret basic matrix concepts (identity, inverse).

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
import scipy.stats as stats

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
df = pd.DataFrame({'x': np.random.rand(100), 'y': np.random.rand(100)})
print("Covariance:\n", df.cov())
print("Correlation:\n", df.corr())
```

#### Visualizing with a Heatmap

```python
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
# Bernoulli
from scipy.stats import bernoulli
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

## Mathematical Notation Reference

| Symbol     | Name/Meaning                              | Plain English                | Python Equivalent            |
|------------|-------------------------------------------|------------------------------|------------------------------|
| $Σ$        | Sigma, summation                          | Add up a sequence            | `sum()`                      |
| $Π$        | Pi, product                               | Multiply a sequence          | `np.prod()`                  |
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
- Try using Scipy's `curve_fit` to fit a curve to noisy data.
- Write a function that computes the variance and standard deviation *by hand* (no NumPy).

---

*Next steps:*  
- Keep practicing! Math is a skill—use it regularly and it will become second nature.
- Ready to move on? Check out the next lesson for practical machine learning.

---