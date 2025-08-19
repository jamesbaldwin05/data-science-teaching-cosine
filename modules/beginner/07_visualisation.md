# Visualisation with Matplotlib (Basic)

Matplotlib is the standard Python library for creating basic visualisations—plots, charts, and graphs. It is highly customizable and integrates well with NumPy and Pandas. While there are higher-level libraries (like seaborn), Matplotlib is the foundation and should be mastered first.

---

## Table of Contents
1. [Matplotlib Basics](#matplotlib-basics)
    - [Importing Matplotlib](#importing-matplotlib)
    - [The Figure and Axes](#the-figure-and-axes)
    - [Plotting Data](#plotting-data)
2. [Line Plots](#line-plots)
3. [Scatter Plots](#scatter-plots)
4. [Histograms](#histograms)
5. [Bar Plots](#bar-plots)
6. [Customising Plots](#customising-plots)
    - [Titles and Labels](#titles-and-labels)
    - [Legends](#legends)
    - [Ticks](#ticks)
    - [Colors, Linestyles, Markers](#colors-linestyles-markers)
7. [Saving Figures](#saving-figures)
8. [Subplots and Multiple Plots](#subplots-and-multiple-plots)

---

## Matplotlib Basics

### Importing Matplotlib

```python
import matplotlib.pyplot as plt
```

The convention is to import pyplot as `plt`.

### The Figure and Axes

- **Figure**: The entire window or page the plot appears on.
- **Axes**: The individual plot or graph (a figure can have multiple axes).

You can create a figure and axes explicitly, but most simple plots use the default:

```python
plt.figure()
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
```

Or the modern, more flexible way:

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
```

---

## Plotting Data

Matplotlib works with lists, NumPy arrays, or Pandas Series as data sources.

---

## Line Plots

The most basic plot—good for showing trends over continuous data.

```python
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```

---

## Scatter Plots

For visualising relationships between two variables.

```python
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.title("Random Scatter")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()
```

---

## Histograms

For showing the distribution of data.

```python
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

---

## Bar Plots

For categorical data.

```python
categories = ['A', 'B', 'C']
values = [5, 7, 3]
plt.bar(categories, values)
plt.title("Bar Plot Example")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()
```

---

## Customising Plots

### Titles and Labels

```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("My Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
```

### Legends

```python
plt.plot([1, 2, 3], [1, 4, 9], label="Squares")
plt.plot([1, 2, 3], [1, 2, 3], label="Linear")
plt.legend()
plt.show()
```

### Ticks

```python
plt.plot([0, 1, 2, 3], [0, 1, 4, 9])
plt.xticks([0, 1, 2, 3], ['zero', 'one', 'two', 'three'])
plt.yticks([0, 1, 4, 9])
plt.show()
```

### Colors, Linestyles, Markers

```python
plt.plot([1, 2, 3], [4, 5, 6], color='red', linestyle='--', marker='o')
plt.show()
```

---

## Saving Figures

```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("my_figure.png")    # Saves as PNG
plt.savefig("my_figure.pdf")    # Saves as PDF
plt.close() # Close the figure
```

---

## Subplots and Multiple Plots

You can create multiple plots in one figure.

```python
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

x = np.linspace(0, 2*np.pi, 100)
axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title("Sine")

axs[0, 1].plot(x, np.cos(x))
axs[0, 1].set_title("Cosine")

axs[1, 0].scatter(np.random.rand(50), np.random.rand(50))
axs[1, 0].set_title("Scatter")

axs[1, 1].hist(np.random.randn(1000), bins=30)
axs[1, 1].set_title("Histogram")

plt.tight_layout()
plt.show()
```

---

Matplotlib offers much more—see the [official documentation](https://matplotlib.org/stable/contents.html) for advanced customisation options.