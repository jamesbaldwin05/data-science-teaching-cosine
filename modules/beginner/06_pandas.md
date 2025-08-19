# Pandas

Pandas is a powerful and flexible open-source data analysis and manipulation library for Python. It is built on top of NumPy and is designed for working with structured data, such as tabular, time series, and matrix data. Pandas makes it easy to clean, analyze, and visualize data, making it an essential tool for data scientists and analysts.

---

## Table of Contents

1. [Introduction to Pandas](#introduction-to-pandas)
2. [Pandas Basics](#pandas-basics)
    - [Importing pandas](#importing-pandas)
    - [Series and DataFrame](#series-and-dataframe)
    - [Creating Series and DataFrames](#creating-series-and-dataframes)
    - [Reading Data from Files](#reading-data-from-files)
3. [DataFrame Attributes and Methods](#dataframe-attributes-and-methods)
    - [Shape, Columns, Dtypes](#shape-columns-dtypes)
    - [Info, Head, Tail, Describe](#info-head-tail-describe)
    - [Value Counts](#value-counts)
4. [Indexing & Selection](#indexing--selection)
    - [.loc and .iloc](#loc-and-iloc)
    - [Boolean Indexing](#boolean-indexing)
    - [at, iat, and Chained Indexing](#at-iat-and-chained-indexing)
5. [Operations](#operations)
    - [Arithmetic and Broadcasting](#arithmetic-and-broadcasting)
    - [String Methods](#string-methods)
    - [Datetime Methods](#datetime-methods)
6. [Handling Missing Data](#handling-missing-data)
    - [Detecting Missing Data](#detecting-missing-data)
    - [Filling Missing Data](#filling-missing-data)
    - [Dropping Missing Data](#dropping-missing-data)
    - [Interpolation](#interpolation)
7. [GroupBy Operations](#groupby-operations)
    - [Aggregation](#aggregation)
    - [Transform and Filter](#transform-and-filter)
8. [Merging & Joining](#merging--joining)
    - [merge, join, concat, append](#merge-join-concat-append)
9. [Reshaping DataFrames](#reshaping-dataframes)
    - [Pivot, Pivot Table, Melt](#pivot-pivot-table-melt)
    - [Stack and Unstack](#stack-and-unstack)
10. [Time Series and Dates](#time-series-and-dates)
    - [Datetime Index and to_datetime](#datetime-index-and-to_datetime)
    - [Resampling, Rolling, and Shifting](#resampling-rolling-and-shifting)
11. [Input/Output Operations](#inputoutput-operations)
    - [Reading and Writing CSV](#reading-and-writing-csv)
    - [Excel and JSON](#excel-and-json)
12. [Conclusion](#conclusion)

---

## Pandas Basics

### Importing pandas

Pandas is conventionally imported as `pd`:

```python
# no-run
import pandas as pd
```

It is de facto standard to use `pd` as an alias for Pandas. As with the last lesson, examples beyond this point will not show this.

---

### Series and DataFrame

- **Series**: A one-dimensional labeled array for any data type.
- **DataFrame**: A two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).

---

### Creating Series and DataFrames

#### From Lists

```python
# Series from a list
s = pd.Series([1, 3, 5, 7])
print(s)
```

```python
# DataFrame from a list of lists
df = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
print(df)
```

#### From Dictionaries

```python
# Series from a dictionary
s = pd.Series({'a': 10, 'b': 20})
print(s)
```

```python
# DataFrame from a dictionary
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
print(df)
```

#### From Arrays

```python
import numpy as np
arr = np.array([[1, 2], [3, 4]])
df = pd.DataFrame(arr, columns=['Col1', 'Col2'])
print(df)
```

#### Built-in Constructors

- `pd.Series(data, index=...)`
- `pd.DataFrame(data, columns=..., index=...)`
- `pd.read_csv('file.csv')`
- `pd.read_excel('file.xlsx')`
- `pd.read_json('file.json')`

---

### Reading Data from Files

```python
# Read CSV
df = pd.read_csv('data.csv')

# Read Excel
df = pd.read_excel('data.xlsx')

# Read JSON
df = pd.read_json('data.json')
```

---

## DataFrame Attributes and Methods

### Shape, Columns, Dtypes

```python
print(df.shape)        # (rows, columns)
print(df.columns)      # Column names (Index object)
print(df.dtypes)       # Data types of columns
```

---

### Info, Head, Tail, Describe

```python
df.info()              # Summary of DataFrame
print(df.head())       # First 5 rows
print(df.tail(3))      # Last 3 rows
print(df.describe())   # Summary statistics (numeric columns)
```

---

### Value Counts

```python
print(df['column_name'].value_counts())
```

---

## Indexing & Selection

### .loc and .iloc

- `.loc` - label-based selection
- `.iloc` - integer position-based selection

```python
# By label
print(df.loc[0, 'A'])

# By integer index
print(df.iloc[0, 1])

# Slicing rows
print(df.loc[2:4])
print(df.iloc[2:5])
```

---

### Boolean Indexing

```python
# Rows where column 'age' > 25
print(df[df['age'] > 25])
```

---

### at, iat, and Chained Indexing

- `.at` and `.iat` are faster accessors for single values.

```python
value = df.at[0, 'A']
value = df.iat[0, 1]
```

**Warning:** Avoid chained indexing (e.g., `df[df['A'] > 1]['B']`). Use `.loc` to avoid unexpected behavior.

---

## Operations

### Arithmetic and Broadcasting

```python
# Add 10 to all elements
df2 = df + 10

# Element-wise addition
df3 = df1 + df2

# Operations with alignment by index/column
```

---

### String Methods

Pandas has vectorized string methods (use `.str` accessor):

```python
df['name'].str.upper()
df['email'].str.contains('@gmail.com')
df['city'].str.replace('York', 'Town')
```

---

### Datetime Methods

Use `.dt` accessor for datetime operations:

```python
df['date'] = pd.to_datetime(df['date'])
df['date'].dt.year
df['date'].dt.month
df['date'].dt.weekday
```

---

## Handling Missing Data

### Detecting Missing Data

```python
df.isna()          # DataFrame of booleans
df.isnull()        # Same as isna()
df.notna()
```

---

### Filling Missing Data

```python
df_filled = df.fillna(0)
df['col'] = df['col'].fillna(df['col'].mean())
```

---

### Dropping Missing Data

```python
df_no_na = df.dropna()                  # Drop rows with any missing values
df_no_na = df.dropna(axis=1)            # Drop columns with any missing values
df_no_na = df.dropna(subset=['col1'])   # Drop rows where col1 is NA
```

---

### Interpolation

```python
df.interpolate(method='linear')
```

---

## GroupBy Operations

### Aggregation

```python
grouped = df.groupby('category')
print(grouped['value'].mean())
print(grouped.agg({'value': ['mean', 'sum']}))
```

---

### Transform and Filter

```python
# Transform
df['demeaned'] = df.groupby('group')['value'].transform(lambda x: x - x.mean())

# Filter groups
filtered = df.groupby('group').filter(lambda x: len(x) > 2)
```

---

## Merging & Joining

### merge, join, concat, append

```python
# merge
pd.merge(df1, df2, on='key', how='inner')

# join
df1.join(df2.set_index('key'), on='key')

# concat
pd.concat([df1, df2], axis=0)   # Stack rows
pd.concat([df1, df2], axis=1)   # Side by side

# append (deprecated in pandas 2.0, use concat)
df1 = pd.concat([df1, df2])
```

---

## Reshaping DataFrames

### Pivot, Pivot Table, Melt

```python
# Pivot
pivoted = df.pivot(index='date', columns='city', values='value')

# Pivot table with aggregation
pd.pivot_table(df, values='value', index='date', columns='city', aggfunc='mean')

# Melt (wide to long)
melted = df.melt(id_vars=['id'], value_vars=['A', 'B'])
```

---

### Stack and Unstack

```python
stacked = df.stack()      # Columns to rows (creates MultiIndex)
unstacked = stacked.unstack()  # Rows to columns
```

---

## Time Series and Dates

### Datetime Index and to_datetime

```python
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
print(df.index)
```

---

### Resampling, Rolling, and Shifting

```python
# Resample to monthly and average
df.resample('M').mean()

# Rolling window
df['value'].rolling(window=3).mean()

# Shift data
df['yesterday'] = df['value'].shift(1)
```

---

## Input/Output Operations

### Reading and Writing CSV

```python
# Read CSV
df = pd.read_csv('data.csv')

# Write CSV
df.to_csv('output.csv', index=False)
```

---

### Excel and JSON

```python
# Read Excel
df = pd.read_excel('file.xlsx')

# Write Excel
df.to_excel('output.xlsx', index=False)

# Read JSON
df = pd.read_json('data.json')

# Write JSON
df.to_json('output.json')
```

---

## Conclusion

Pandas is a crucial tool for data manipulation, exploration, and analysis in Python. With its rich set of features and intuitive syntax, you can clean, analyze, and visualize complex datasets efficiently. Practice these basics, and you'll be ready to tackle real-world data problems!

---