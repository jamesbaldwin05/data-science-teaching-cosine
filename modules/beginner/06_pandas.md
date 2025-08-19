# Pandas (Python Data Analysis Library)
Pandas is a powerful Python library for data manipulation, analysis, and cleaning. It provides easy-to-use data structures—**Series** and **DataFrame**—designed for working with structured, tabular data. Pandas is widely used for:
- Importing/exporting data (CSV, Excel, SQL, etc.)
- Filtering, selecting, and transforming data
- Handling missing data
- Aggregation and group operations
- Merging and joining datasets

---

## Table of Contents
1. [Pandas Basics](#pandas-basics)
    - [Importing Pandas](#importing-pandas)
    - [Series and DataFrames](#series-and-dataframes)
    - [Creating DataFrames](#creating-dataframes)
    - [Reading/Writing Data](#readingwriting-data)
2. [Viewing and Inspecting Data](#viewing-and-inspecting-data)
3. [Indexing and Selecting Data](#indexing-and-selecting-data)
    - [Selecting Columns](#selecting-columns)
    - [Selecting Rows](#selecting-rows)
    - [Boolean Indexing](#boolean-indexing)
    - [Setting Values](#setting-values)
4. [Missing Data](#missing-data)
5. [Basic Operations](#basic-operations)
    - [Descriptive Statistics](#descriptive-statistics)
    - [Aggregation](#aggregation)
6. [Sorting and Ranking](#sorting-and-ranking)
7. [GroupBy: Split-Apply-Combine](#groupby-split-apply-combine)
8. [Merging and Joining DataFrames](#merging-and-joining-dataframes)
9. [Reshaping Data](#reshaping-data)
    - [Pivot](#pivot)
    - [Melt](#melt)
    - [Stack/Unstack](#stackunstack)
10. [Exporting Data](#exporting-data)

---

## Pandas Basics

Pandas is built on top of NumPy and provides high-level data structures for tabular data, making data analysis tasks much simpler.

### Importing Pandas

```python
import pandas as pd
```
The convention is to import pandas as `pd`.

### Series and DataFrames

- **Series**: a one-dimensional labeled array (like a column of a spreadsheet).
- **DataFrame**: a two-dimensional table with labeled axes (rows and columns).

#### Creating a Series

```python
s = pd.Series([10, 20, 30, 40])
print(s)
```

You can specify an index:
```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
```

#### Creating a DataFrame

From a dictionary:
```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)
print(df)
```

From a list of dicts:
```python
records = [
    {'Name': 'Alice', 'Age': 24},
    {'Name': 'Bob', 'Age': 27}
]
df = pd.DataFrame(records)
print(df)
```

### Reading/Writing Data

Pandas makes it easy to read from and write to many formats.

```python
df = pd.read_csv('data.csv')      # Read CSV
df = pd.read_excel('data.xlsx')   # Read Excel
df = pd.read_json('data.json')    # Read JSON
```

To write:
```python
df.to_csv('output.csv', index=False)      # Write CSV
df.to_excel('output.xlsx', index=False)   # Write Excel
df.to_json('output.json')                 # Write JSON
```

---

## Viewing and Inspecting Data

- `.head(n)` shows the first n rows (default 5)
- `.tail(n)` shows the last n rows
- `.shape` returns tuple (rows, columns)
- `.info()` summary of DataFrame
- `.describe()` basic statistics

```python
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
```

---

## Indexing and Selecting Data

Selecting data by columns, rows, or conditions.

### Selecting Columns

```python
df['Name']        # single column as Series
df[['Name', 'Age']]  # multiple columns as DataFrame
```

### Selecting Rows

- **By integer position**: `.iloc[]`
- **By label/index**: `.loc[]`

```python
df.iloc[0]             # first row
df.iloc[1:3]           # rows at positions 1 and 2

df.loc[0]              # row with index label 0
df.loc[[0, 2]]         # rows with index labels 0 and 2
```

### Boolean Indexing

```python
df[df['Age'] > 23]     # rows where Age > 23
```

### Setting Values

```python
df.loc[0, 'Age'] = 25
df['NewCol'] = df['Age'] * 2
```

---

## Missing Data

Pandas represents missing data as `NaN`.

- `.isna()` or `.isnull()` to check for missing values
- `.fillna(value)` to replace missing values
- `.dropna()` to remove missing values

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
print(df.isna())
df_filled = df.fillna(0)
df_dropped = df.dropna()
```

---

## Basic Operations

### Descriptive Statistics

- `.sum()`, `.mean()`, `.max()`, `.min()`, `.std()`, `.var()`, `.count()`, `.median()`
- By default, operates column-wise

```python
print(df['Age'].mean())
print(df.mean())
```

### Aggregation

You can aggregate by column or row.

```python
df.agg({'Age': ['mean', 'min', 'max']})
```

---

## Sorting and Ranking

- `.sort_values(by, ascending=True)` sorts by a column
- `.sort_index()` sorts by index

```python
df_sorted = df.sort_values('Age')
```

---

## GroupBy: Split-Apply-Combine

Group rows by a column and aggregate.

```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar'],
                   'B': [1, 2, 3, 4]})

grouped = df.groupby('A').sum()
print(grouped)
```

You can also use multiple aggregations:

```python
df.groupby('A').agg({'B': ['mean', 'max']})
```

---

## Merging and Joining DataFrames

- `pd.merge(left, right, on, how)` merges DataFrames (like SQL joins)
- `.join()` joins on indexes

```python
df1 = pd.DataFrame({'key': ['a', 'b', 'c'], 'val1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'val2': [4, 5, 6]})

merged = pd.merge(df1, df2, on='key', how='inner')
print(merged)
```

---

## Reshaping Data

### Pivot

```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar'],
                   'B': ['one', 'two', 'one'],
                   'C': [1, 2, 3]})

pivoted = df.pivot(index='A', columns='B', values='C')
print(pivoted)
```

### Melt

Unpivot a DataFrame from wide to long format.

```python
df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
melted = pd.melt(df)
print(melted)
```

### Stack/Unstack

Change between "long" and "wide" formats.

```python
df = pd.DataFrame({'A': ['foo', 'bar'], 'B': [1, 2]})
stacked = df.stack()
print(stacked)
unstacked = stacked.unstack()
print(unstacked)
```

---

## Exporting Data

```python
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
df.to_json('output.json')
```

---

Pandas is a comprehensive library with many more features; consult the [official documentation](https://pandas.pydata.org/pandas-docs/stable/) for advanced usage.