# Visualisation with Matplotlib

Matplotlib is the standard Python library for creating basic plots, charts, and graphs. It is highly customizable and integrates well with NumPy arrays and Pandas DataFrames.

[Official Documentation](https://matplotlib.org/stable/contents.html)

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
# no-run
import matplotlib.pyplot as plt
```

It is de facto standard to use `plt` for Matplotlib.

### The Figure and Axes

- **Figure**: The entire window or page the plot appears on.
- **Axes**: The individual plot or graph (a figure can have multiple axes).

You can create a figure and axes explicitly, but most simple plots use the default:

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.figure()
plt.plot(x,y)
plt.show()
```

The more modern way to create a figure and axes is with the `plt.subplots()` function. This will be covered in a later lesson but is shown here as an example.

```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.show()
```

---

## Plotting Data

### Line Plots

`plt.plot()` creates a line plot, the most basic plot—good for showing trends over continuous data.

```python
import numpy as np
x = np.linspace(0, 10, 100)      # 100 evenly spaced numbers between 0 and 10
y = np.sin(x)

plt.plot(x, y)
plt.show()
```

### Scatter Plots

`plt.scatter()` creates a scatter plot, used for visualising relationships between two variables.

```python
import numpy as np
x = np.random.rand(50)         # 50 random numbers between 0 and 1
y = np.random.rand(50)

plt.scatter(x, y)
plt.show()
```

### Histograms

`plt.hist()` creates a histogram, used for showing the distribution of data.

```python
import numpy as np
data = np.random.randn(1000)     # 1000 random numbers from a standardised normal distribution

plt.hist(data, bins=30)          # data is split into 30 evenly sized intervals
plt.show()
```

### Bar Plots

`plt.bar()` creates a bar plot, used for categorical data.

```python
categories = ['A', 'B', 'C']
values = [5, 7, 3]

plt.bar(categories, values)
plt.show()
```

### Box Plots

`plt.boxplot()` creates a boxplot, used for showing statistical features of data.

```python
import numpy as np
data = np.random.randn(100)  # 100 samples from a normal distribution

plt.boxplot(data)
plt.show()
```

### Pie Charts

`plt.pie()` creates a pie chart, used for showing relative proportions.

```python
sizes = [30, 25, 20, 15, 10]   # percentages or values
labels = ["A", "B", "C", "D", "E"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.show()
```

---

## Customising Plots

### Titles and Labels

We can easily add titles and labels to graphs with Matplotlib, using the functions `plt.title()`, `plt.xlabel` and `plt.ylabel`.

```python
plt.plot([1, 2, 3], [4, 5, 6])

plt.title("My Plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()
```

### Legends

A legend is a little box with information about a graph, allowing multiple graphs to be displayed at once without confusion. To use it, add the `label` argument to your plots and call the function `plt.legend()`.

```python
import numpy as np

x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x), label="Sine wave")
plt.plot(x, np.cos(x), label="Cosine wave")

plt.title("Trigonometric Functions")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.show()
```

### Ticks

Ticks are the marks (and their labels) along the axes. They can be used to highlight important points in the data (for example, the points where two plots cross). They are changed with `plt.xticks()` and `plt.yticks()`.

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)

plt.xticks([0, 5])
plt.yticks([0, 5, 10])

plt.show()
```

### Limits

To change how much of an axis is visible, we can use `plt.xlim()` and `plt.ylim()`.

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)

plt.xlim(3, 6)
plt.ylim(0, 12)

plt.show()
```


### Colours, Linestyles, Markers

There are a variety of colours, linestyles and markers which can be used in MatPlotLib. These are all changed by passing `color`, `linestyle` and `marker` arguments into the plot function.

```python
plt.plot([1, 2, 3], [4, 5, 6], color='red', linestyle='--', marker='o')
plt.show()
```

They can also be passed as one shorthand parameter for the same result. For example, `plt.plot([1, 2, 3], [4, 5, 6], 'ro--')` for the above example (r for red color, o for o marker and -- for linestyle).

### Change figure size

To change the size of the figure (plot), we can use `plt.figure(figsize=(w,h))` which takes width and height as a tuple, measured in inches.

```python
import numpy as np

x = np.linspace(0, 10, 100)

plt.figure(figsize=(8,2))
plt.plot(x, np.sin(x))
plt.show()
```

---

## Saving Figures

It is easy to save a plot with MatPlotLib using `plt.savefig()`. Plots can be saved in many formats such as `.jpg`, `.png`, `.svg`, `.pdf` etc. and these are specified in the name of the file passed to the function. Always use `plt.savefig()` before `plt.show()` since the plot may be cleared depending on your environment.

```python
# no-run
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("my_figure.png")    # Saves as PNG
plt.savefig("my_figure.pdf")    # Saves as PDF
```

---

## Subplots and Multiple Plots

You can create multiple plots in one figure using `plt.subplot()`. It takes 3 parameters:
- Rows (how many plots should be displayed alongside each other vertically).
- Columns (how many plots should be displayed alongside each other horizontally)
- Plot number (index assigned row-wise to each subplot)

```python
plt.subplot(2, 1, 1)   # 2 rows, 1 column, first plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("First Plot")

# Second subplot
plt.subplot(2, 1, 2)   # 2 rows, 1 column, second plot
plt.plot([1, 2, 3], [1, 2, 3])
plt.title("Second Plot")

plt.tight_layout()
plt.show()
```

These can actually all be passed as one parameter which is automatically split into rows, columns and plot. For example, `plt.subplot(312)` for the second plot of a subplot with 3 rows and 1 column.

---