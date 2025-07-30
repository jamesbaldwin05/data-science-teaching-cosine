# Lesson 3: Essential Mathematics for Data Science

Welcome! This lesson will guide you through the core mathematics needed for data science. No advanced math background required—just a willingness to learn and basic programming skills.

---

## 1. Linear Algebra: Foundation of Data

**Concept**  
Linear algebra lets us represent and manipulate data in tables, images, and models. At its heart are vectors (lists of numbers) and matrices (tables of numbers).

### Vectors

A vector is simply an ordered list of numbers.  
Example: A flower’s measurements `[5.1, 3.5, 1.4, 0.2]`.

```python
import numpy as np

v = np.array([5.1, 3.5, 1.4, 0.2])
print("Vector:", v)
```

### Matrices

A matrix is a collection of vectors (rows or columns).  
Example: Multiple flowers’ measurements.

```python
M = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2]
])
print("Matrix:\n", M)
```

---

### Matrix Operations

#### Matrix Multiplication

Combine matrices to transform data or apply weights in machine learning.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
C = A @ B  # or np.matmul(A, B)
print("A x B =\n", C)
```

#### Dot Product

A way to combine two vectors. Used in similarity, projections, and neural networks.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)
print("Dot product:", dot)
```

#### Transpose

Flip rows and columns, useful for aligning data.

```python
print("Transpose:\n", M.T)
```

#### Inverse

The “undo” operation for square matrices (like dividing for numbers).  
Not all matrices have inverses.

```python
square = np.array([[1, 2], [3, 4]])
inv = np.linalg.inv(square)
print("Inverse:\n", inv)
```

#### Eigenvalues & Eigenvectors

Capture the “main directions” of data—key for methods like Principal Component Analysis (PCA).

```python
eigvals, eigvecs = np.linalg.eig(square)
print("Eigenvalues:", eigvals)
print("Eigenvectors:\n", eigvecs)
```

#### Application: PCA (Principal Component Analysis)

PCA finds the “axes” along which your data varies the most.  
It’s used for reducing dimensions while preserving important information.

```python
from sklearn.decomposition import PCA

X = np.random.rand(100, 5)  # 100 samples, 5 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print("Reduced data shape:", X_reduced.shape)
```

---

### Exercise

"""
Create a 3x3 matrix in NumPy and calculate its transpose and inverse (if possible).
"""

---

## 2. Statistics: Understanding Data

**Concept**  
Statistics helps us summarize, visualize, and make inferences from data.

### Descriptive Statistics

| Statistic         | What it means               | Python Example         |
|-------------------|----------------------------|-----------------------|
| Mean              | Average value               | `np.mean(arr)`        |
| Median            | Middle value                | `np.median(arr)`      |
| Mode              | Most frequent value         | `scipy.stats.mode(arr)`|
| Variance          | Spread from the mean        | `np.var(arr)`         |
| Standard Deviation| Typical distance from mean  | `np.std(arr)`         |

```python
import numpy as np
from scipy import stats

arr = np.array([1, 2, 2, 3, 4])
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Mode:", stats.mode(arr, keepdims=True).mode[0])
print("Variance:", np.var(arr))
print("Standard Deviation:", np.std(arr))
```

#### Visual Aids

- **Histogram**: Shows how data is distributed (like a bar chart for numbers).
- **Boxplot**: Shows median, quartiles, and outliers.

```python
import matplotlib.pyplot as plt

plt.hist(arr)
plt.title("Histogram")
plt.show()

plt.boxplot(arr)
plt.title("Boxplot")
plt.show()
```

---

### Probability Distributions

- **Normal (Gaussian)**: Bell-curve, many natural phenomena.
- **Binomial**: Number of successes in repeated yes/no events.
- **Uniform**: All values equally likely.

```python
# Generate and plot a normal distribution
data = np.random.normal(loc=0, scale=1, size=1000)
plt.hist(data, bins=30, density=True)
plt.title("Normal Distribution")
plt.show()
```

---

### Sampling

Draw random samples to estimate properties of a larger population.

```python
population = np.arange(1000)
sample = np.random.choice(population, size=100, replace=False)
print("Sample mean:", np.mean(sample))
```

---

### Hypothesis Testing

A way to test if a result is likely due to chance.

- **Null hypothesis (H0):** No effect or difference.
- **Alternative hypothesis (H1):** There is an effect.
- **p-value:** Probability of seeing the data if H0 is true.  
  (If p < 0.05, we often reject H0.)

```python
# Example: Does group A have higher average than group B?
group_a = np.random.normal(0, 1, 100)
group_b = np.random.normal(0.5, 1, 100)
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(group_a, group_b)
print("p-value:", p_val)
```

---

### Confidence Intervals

A range where we expect the true value to lie, with a certain probability.

```python
import scipy.stats as stats

mean = np.mean(group_a)
sem = stats.sem(group_a)  # standard error
conf_int = stats.t.interval(0.95, len(group_a)-1, loc=mean, scale=sem)
print("95% confidence interval:", conf_int)
```

---

### Correlation

Measures how two variables move together.

```python
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)
corr = np.corrcoef(x, y)[0, 1]
print("Correlation:", corr)
```

---

### Exercise

"""
Given two lists of numbers, calculate and plot their correlation using NumPy and matplotlib.
"""

---

## 3. Probability: Reasoning Under Uncertainty

**Concept**  
Probability helps us model uncertainty—core to predicting and learning from data.

### Basic Rules

- **Probability of an event:** Between 0 (impossible) and 1 (certain).
- **Complement:** P(not A) = 1 - P(A)
- **Addition Rule:** P(A or B) = P(A) + P(B) - P(A and B)
- **Multiplication Rule:** P(A and B) = P(A) * P(B) if A and B are independent.

### Conditional Probability

“What's the chance of A, given that B happened?”  
Notation: P(A | B) = P(A and B) / P(B)

```python
# Suppose 60% of emails are spam, and 10% of all emails contain "free" and are spam.
p_spam = 0.6
p_free_and_spam = 0.1
p_free_given_spam = p_free_and_spam / p_spam
print("P('free' | spam):", p_free_given_spam)
```

### Independence

A and B are independent if knowing B doesn't change the chance of A.

```python
# Rolling two dice
```

### Key Distributions

#### Discrete

- **Binomial:** Number of successes in n trials.
- **Poisson:** Number of events in fixed time/space.

```python
from scipy.stats import binom, poisson

# Binomial: 10 coin flips, prob of 3 heads
print("Binomial P(X=3):", binom.pmf(3, n=10, p=0.5))

# Poisson: Probability of 2 events with average rate 4
print("Poisson P(X=2):", poisson.pmf(2, mu=4))
```

#### Continuous

- **Normal (Gaussian)**
- **Exponential**: Time between events

```python
from scipy.stats import norm, expon

print("Normal P(X<1):", norm.cdf(1, loc=0, scale=1))
print("Exponential P(X<2):", expon.cdf(2, scale=1))
```

---

### Exercise

"""
Simulate flipping a fair coin 100 times in Python. How many heads did you get?
"""

---

## 4. Calculus (for Optimization)

**Concept**  
Calculus helps us optimize models—find the “best” parameters by minimizing loss.

### Derivatives

A derivative measures how a function changes (“slope”).  
In machine learning, we use derivatives to update weights.

```python
# Slope of f(x) = x**2 at x = 3
def f(x):
    return x**2

x = 3
slope = 2 * x  # derivative of x^2 is 2x
print("Slope at x=3:", slope)
```

### Gradients

A gradient is a vector of partial derivatives—tells us how to change each input to increase or decrease the output.

```python
# For f(x, y) = x^2 + y^2
def grad(x, y):
    return np.array([2 * x, 2 * y])

print("Gradient at (2,3):", grad(2, 3))
```

### Partial Derivatives

Change one variable at a time, holding others fixed.

---

### Application: Gradient Descent

An algorithm to find minima (or maxima) by following the negative gradient.

```python
# Find minimum of f(x) = (x-3)^2
x = 0
for i in range(10):
    grad = 2 * (x - 3)
    x -= 0.1 * grad
    print(f"Step {i}: x = {x:.2f}")
```

---

### Exercise

"""
Write Python code to compute the derivative of f(x) = 2x^3 at x=4, both analytically and by finite difference (numerical approximation).
"""

---

## 5. Mathematical Notation: Reading the Language

Here are some symbols you'll see often:

| Symbol | Name              | Meaning                         | Example                        |
|--------|-------------------|---------------------------------|--------------------------------|
| Σ      | Summation         | Add up terms                    | Σxᵢ (sum all x's)              |
| Π      | Product           | Multiply terms                  | Πxᵢ (multiply all x's)         |
| ∇      | Gradient          | Vector of derivatives           | ∇f(x) (gradient of f at x)     |
| μ      | Mu                | Mean                            | μ = mean(x)                    |
| σ      | Sigma             | Standard deviation              | σ = std(x)                     |
| P(A)   | Probability       | Chance of event A               | P(X > 5)                       |
| E[X]   | Expectation       | Average value of X              | E[X] = mean(X)                 |

---

### Example: Summation in Python

```python
arr = np.array([1, 2, 3, 4])
print("Σxᵢ =", np.sum(arr))
print("Πxᵢ =", np.prod(arr))
```

---

### Quiz

**Question:**  
What does “∇f(x)” mean in machine learning?

- A) The sum of all x
- B) The product of all x
- C) The gradient (vector of partial derivatives)
- D) The average value of x

**Answer:** C

---

## Summary

You've now learned the essential math for data science:
- Linear algebra (vectors, matrices, PCA)
- Statistics (descriptive stats, distributions, hypothesis testing)
- Probability (basic rules, conditional, key distributions)
- Calculus (derivatives, gradients, optimization)
- Mathematical notation (read research and docs with confidence)

Practice the exercises above, and use the code snippets as templates for your own data science projects!

---