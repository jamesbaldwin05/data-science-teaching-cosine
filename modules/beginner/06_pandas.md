# Pandas
Pandas is a powerful Python library for data manipulation, analysis, and cleaning. It provides easy-to-use data structures—**Series** and **DataFrame**—designed for working with structured, tabular data. Pandas is widely used for:
- Importing/exporting data (CSV, Excel, SQL, etc.)
- Filtering, selecting, and transforming data
- Handling missing data
- Aggregation and group operations
- Merging and joining datasets

[Official Documentation](https://pandas.pydata.org/docs/)

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

To import the Pandas library, use the code below:

```python
# no-run
import pandas as pd
```
It is de facto standard to use `pd` as an alias for Pandas.

Similar to last lesson, every example beyond this point will not show the code to import the library.

### Series and DataFrames

- **Series**: a one-dimensional labeled array (like a column of a spreadsheet).
- **DataFrame**: a two-dimensional table with labeled axes (rows and columns).

### Creating a series:

```python
s = pd.Series([10, 20, 30, 40])
print(s)
```

You can specify an index:
```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s)
```

### Creating a DataFrame:

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

To read:

```python
# no-run
df = pd.read_csv('data.csv')      # Read CSV
df = pd.read_excel('data.xlsx')   # Read Excel
df = pd.read_json('data.json')    # Read JSON
```

To write:

```python
# no-run
df.to_csv('output.csv', index=False)      # Write CSV
df.to_excel('output.xlsx', index=False)   # Write Excel
df.to_json('output.json')                 # Write JSON
```

---

## Basic Attributes and Methods

- `.shape` returns the number of rows and columns in the DataFrame, in the form `(rows, columns)`.

- `.size` returns the total number of elements (`rows x columns`).

- `.columns` returns the labels of all columns.

- `.dtypes` returns the data type of each column.

- `.head(n)` or `.tail(n)` shows the first or last `n` rows (default is 5).

- `.sample(n)` returns a random sample of rows.

- `.info()`returns a summary of DataFrame.

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data)

print(df.dtypes)
```

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data)

print(df.head())
```

---

## Indexing and Selecting Data

### Selecting Columns

- By using `data["Column"]` we can access columns from within a DataFrame using a similar style of indexing as native python dictionaries. This is returned as a Series object.

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

print(df["Age"])
```

- If we want to keep the result as a DataFrame (instead of a Series), we can do `data[["Column"]]`. This also allows us to access multiple columns by passing a list of columns such as `data[["Column1", "Column2"]]`.

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data)

print(df[["Name", "Age"]])
```

### Selecting Rows

To find a row using Pandas, you can either use its integer position, or its given label/index.

**By integer position**: To select a row using its integer position within the DataFrame, use `.iloc[]`. We can use any of the following methods:
- `df.iloc[0]` returns the first row.
- `df.iloc[-1]` returns the last row.
- `df.iloc[0:3]` returns the first 3 rows
- `df.iloc[[0, 2, 4]]` returns the first, third and fifth row.
- `df.iloc[0:3, [1,0]]` returns the first 3 rows but only columns 1 and 0.

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data)

print(df.iloc[5:8, [1, 2]])
```

**By label/index**: To select a row using its label/index within the DataFrame, use `.loc[]`.
- `df.loc["A100"]` returns the row with label "A100".
- `df.loc["A100":"A105"]` returns the rows from "A100" to "A105" inclusive.
- `df.loc[["A100", "B100"]]` returns the rows "A100" and "B100".
- `df.loc[["A100", "B100"], ["Name", "Age"]]` returns the rows "A100" and "B100" but only columns "Name" and "Age".

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data, index=range(1000, 1010))

print(df.loc[1001:1002, ["Name", "City"]])
```

### Boolean Indexing

We can use Boolean Indexing to filter DataFrames.

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

print(df[df["Age"] > 23])
```

We can include multiple conditions to make more advanced expressions.

```python
data = {
    "ID": range(1, 11),
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan",
             "Fiona", "George", "Hannah", "Ian", "Julia"],
    "Age": [24, 30, 22, 28, 35, 27, 31, 29, 26, 32],
    "City": ["London", "Paris", "Berlin", "Madrid", "Rome",
             "Lisbon", "Vienna", "Prague", "Athens", "Dublin"]
}
df = pd.DataFrame(data, index=range(1000, 1010))

print(df[(df["Age"] > 23) & ((df["Name"].str.startswith("A")) | df["City"].isin(["London", "Paris", "Rome"]))])
```

---

## Changing Data

### Changing a single cell

To change a specific cell, the fastest way to do this is with `.at` or `.iat`:

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

df.at[0, "Age"] = 30                             # you can use a index/label instead of 0 if one exists
df.iat[1, 0] = "Bill"

print(df)
```

You can also use `.loc` or `.iloc` (e.g. `df.loc[0, "Age"] = 30`) but this is not as fast for single cells.

### Changing a whole column

To uniformly change a whole column, we can just select the column and execute the change.

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

df["Age"] = df["Age"] + 1

print(df)
```

Using `.loc` or `.iloc` we can create more advanced expressions and use Boolean indexing.

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data, index = ["a", "b", "c"])

df.loc[["a", "c"], ["Name", "Age"]] = [["Anna", 26], ["Charles", 23]]

print(df)
```

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

df.loc[df["Name"].str.startswith("B"), "Age"] = 30   # set Age to 30 for all rows where Name starts with "B"

print(df)
```

### Adding a column

To add a new column (or change every row in an existing column) we can use `.assign()` which returns a new DataFrame with the changes.

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [24, 27, 22]}
df = pd.DataFrame(data)

df2 = df.assign(Age=[30, 50, 40], Salary=[50000, 20000, 35000])

print(df2)
```

---

## Missing Data

### Finding missing data
In real datasets, there is often missing or null values, represented as `NaN` in pandas. There are multiple methods to detect and handle this missing data:

- `isna()` or `isnull()` (both the same function) return a DataFrame of the same shape with `True` wherever values are `NaN`.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
print(df.isna())
```

- We can be more specific here and check for columns or rows with any missing values using `.any()`:

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
print(df.isna().any(axis=1))                                     # any rows with missing values are marked as True
```

- `.all()` is a similar function but returns true if **every** value in a row/column is missing.

### Filling missing data

- `.fillna(value)` replaces any missing data with the given value.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
df = df.fillna(0)
print(df)
```

- We can use forward-fill `.ffill()` or backward-fill `.bfill()` to propagate values across rows or columns.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
df = df.ffill()                                              # the value from the previous row is copied for the missing value
print(df)
```

### Removing missing data

- `.dropna()` removes rows or columns containing missing values.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
df_dropped = df.dropna()
print(df_dropped)
```

- Note this method removes rows with missing values by default, use `axis=1` to remove columns with missing values (unlike previous functions where `axis=1` works on rows).

- We can use the `thresh` argument to add a minimum number of values needed for the row to be kept.

```python
df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
df_dropped = df.dropna(thresh=2)                              # keeps rows with at least 2 non-NaN values
print(df_dropped)
```

---

## Basic Operations

### Descriptive Statistics

Pandas comes with various column-wise statistical functions built in. These all ignore missing data (`None` or `NaN`) by default:
- `.sum()` adds up the values.
- `.count()` counts the number of non-NaN entries.
- `.mean()` finds the mean of the values (sum ÷ count of non-null values).
- `.max()` and `.min()` return the maximum and minimum values respectively.
- `.std()` and `.var()` returns the sample standard deviation and sample variance respectively.
- `.median()` returns the middle value when sorted.

```python
df = pd.DataFrame({
    "age": [24, 30, None, 28],
    "score": [90, None, 85, 88],
    "height": [170, 180, 175, None]
})

print(df["height"].mean())
print(df["score"].std())
```

However, they can also be applied to the whole DataFrame at once instead of specific columns. Pandas will then apply them column by column.

```python
df = pd.DataFrame({
    "age": [24, 30, None, 28],
    "score": [90, None, 85, 88],
    "height": [170, 180, 175, None]
})

print(df.sum())
```

These functions can also be applied row-wise if we pass `axis=1`:

```python
df = pd.DataFrame({
    "job1": [24000, 30000, None, 28000],
    "job2": [22000, None, 40000, 18000],
    "job3": [17000, 19000, 17500, None]
})

print(df.mean(axis=1))
```

To get an overall picture of the statistics of a DataFrame, we can use `.describe()`:

```python
df = pd.DataFrame({
    "age": [24, 30, None, 28],
    "score": [90, None, 85, 88],
    "height": [170, 180, 175, None]
})

print(df.describe())
```

We can use `.describe(include="all")` to include non-numeric columns, which will show counts, unique values, top values and frequency.

### Aggregation

Pandas also provides the `.agg()` (or `.aggregate()`) method, which lets you apply multiple functions to multiple columns at once.

```python
df = pd.DataFrame({
    "Age": [24, 30, 22, 28],
    "Score": [90, 85, 88, 92],
    "Height": [170, 180, 175, 178]
})

print(df.agg({'Age': ['mean', 'min', 'max'],
              'Score': ['sum', 'std']}))
```

We can also do this row-wise with `axis=1`:

```python
df = pd.DataFrame({
    "job1": [24000, 30000, None, 28000],
    "job2": [22000, None, 40000, 18000],
    "job3": [17000, 19000, 17500, None]
})

print(df.agg(['sum', 'mean'], axis=1))
```

We can even pass custom functions (lambdas or user defined):

```python
df = pd.DataFrame({
    "Age": [24, 30, 22, 28],
    "Score": [90, 85, 88, 92],
    "Height": [170, 180, 175, 178]
})

print(df.agg({"Age": lambda x: x.max() - x.min()}))
```

---

## Sorting and Ranking

Sorting is often used before plotting or aggregating results to make data more readable.

- `.sort_values(by="column")` sorts by a given column (lowest to highest by default).

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [24, 30, 22, 28],
    "Score": [88, 92, 85, 90]
})

print(df.sort_values(by="Age"))
```

- The argument `ascending=False` sorts them highest to lowest.

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [24, 30, 22, 28],
    "Score": [88, 92, 85, 90]
})

print(df.sort_values(by="Score", ascending=False))
```

- Note these **do not** modify the original DataFrame, unless the argument `inplace=True` is used.

- `.sort_index()` sorts by index

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "ID": [9286, 2031, 3245, 4528],
    "Age": [24, 30, 22, 28],
    "Score": [88, 92, 85, 90]
})
df.set_index("ID", inplace=True)

print(df.sort_index())
```

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "ID": [9286, 2031, 3245, 4528],
    "Age": [24, 30, 22, 28],
    "Score": [88, 92, 85, 90]
})

print(df.sort_index(axis=1))                             # sorts the columns by name
```


- `.rank()` assigns ordinal positions to values and can handle ties in multiple ways. Ranks are 1-based by default, meaning the smallest value gets rank 1 (unless `ascending=False`).

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Score": [88, 92, 85, 92]
})

df["Rank"] = df["Score"].rank()                       # since Bob and Diana have the same score, they receive a rank of 3.5, the average of the 3rd and 4th position.
df["Rank_max"] = df["Score"].rank(method="max")       # they are now both given rank 4 since this is the maximum of the two positions (method=min would give 3)
df["Rank_first"] = df["Score"].rank(method="first")   # ranks are now assigned in order of appearance, so Bob gets 3 and Diana 4
print(df)
```

- We can also quickly access the smallest or largest values based ordered on a column with `.nlargest(n, column)` or `.nsmallest(n, column)`.

```python
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Score": [88, 92, 85, 92]
})

print(df.nlargest(2, "Score"))                        # returns the two rows with the highest score
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

Pandas is a comprehensive library with many more features; consult the [official documentation](https://pandas.pydata.org/pandas-docs/stable/) for advanced usage.