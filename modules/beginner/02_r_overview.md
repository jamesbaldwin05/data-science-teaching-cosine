# R in Data Science Overview

## Who is This Course For?

This course is designed for data analysts, scientists, and learners who want to leverage R for data science. If you have some programming experience (in any language), this module will introduce you to R’s core features and its unique approach to data science—emphasizing vectorization, data frames, and the rich statistical and visualization ecosystem.

---

## Prerequisite R Skills: Self-Assessment Checklist

Before you dive in, check if you can confidently answer "yes" to the following:

- Do you know how to create and manipulate vectors, lists, and data frames?
- Can you subset and index R objects using `[`, `[[`, and `$`?
- Are you familiar with basic arithmetic and logical operations on vectors?
- Have you written and used your own functions?
- Can you install and load R packages with `install.packages()` and `library()`?
- Do you understand the basics of R scripts vs. interactive sessions?
- Have you used the RStudio IDE or the R console?
- Can you read and write CSV files?

If not, consider brushing up on R fundamentals before proceeding.

---

### Where to Brush Up

Need to review R basics? Try these resources:

- [R for Data Science (free online book)](https://r4ds.hadley.nz/)
- [Swirl – Interactive R learning in the console](https://swirlstats.com/)
- [RStudio Primers](https://posit.cloud/learn/primers)
- [Statistical Learning with R (Stanford Online)](https://online.stanford.edu/courses/sohs-ystatslearning-statistical-learning)
- [Quick-R](https://www.statmethods.net/)

---

# Essential R Features & Practices for Data Science

To succeed with R in data science, focus on these language features and workflows.

---

## 1. Vectors & Vectorized Operations

**What/Why:**  
Vectors are R’s fundamental data structure—almost all data in R is built from vectors. Operations on vectors are naturally applied element-wise (“vectorized”), making data transformation concise and fast.

**R vectors vs. Python lists:**  
R vectors are homogeneous (all elements must be the same type), unlike Python lists. For mixed types, use lists.

**Example:**

```r
# Numeric vector and vectorized operations
temps_C <- c(12, 18, 22, 15)
temps_F <- temps_C * 9/5 + 32
print(temps_F)
```

**You should be able to:**  
- Create vectors with `c()`  
- Apply arithmetic, logical, and comparison operations  
- Use `length()`, `sum()`, `mean()`, and vectorized functions

---

## 2. Lists & Data Frames

**What/Why:**  
Lists are flexible containers for heterogeneous data; data frames are list-like structures where each column is a vector (often of different types). Data frames are the standard for tabular data in R.

**Example:**

```r
# List with named elements
person <- list(name = "Alice", scores = c(math = 90, bio = 87))
mean_score <- mean(person$scores)
print(mean_score)

# Data frame
df <- data.frame(name = c("Alice", "Bob"), age = c(25, 30))
df$age[1] <- df$age[1] + 1  # Update value
print(df)
```

**You should be able to:**  
- Create, access, and modify lists and data frames  
- Use `$`, `[[]]`, and `[]` for subsetting  
- Add/remove columns in data frames

---

## 3. Indexing & Subsetting

**What/Why:**  
Selecting and filtering data is key for analysis. R supports flexible indexing: by position, name, or logical condition.

**Example:**

```r
# Subset by position
x <- c(10, 20, 30, 40)
x[2:3]

# Subset by logical condition
ages <- c(15, 22, 18, 30)
adults <- ages[ages >= 18]
print(adults)

# Data frame filtering
df <- data.frame(name = c("Alice", "Bob"), age = c(17, 24))
adults_df <- df[df$age >= 18, ]
print(adults_df)
```

**You should be able to:**  
- Subset vectors and data frames by position, name, or logical vector  
- Filter data frames with conditions  
- Use `which()`, `match()`, and `%in%` for advanced selection

---

## 4. Apply Family Functions (`apply`, `lapply`, `sapply`, `map`)

**What/Why:**  
Apply-family functions provide a succinct way to operate over data structures without explicit loops—crucial for efficient, expressive R code.

**Example:**

```r
# Using lapply to compute lengths of each column
df <- data.frame(a = 1:3, b = c("x", "y", "z"))
lengths <- lapply(df, length)
print(lengths)

# Using sapply for a simplified result
means <- sapply(df[1], mean)
print(means)
```

**You should be able to:**  
- Use `lapply`, `sapply`, `apply` for row/column-wise operations  
- Recognize when to use each  
- Use `purrr::map` for more advanced iteration (in the tidyverse)

---

## 5. Functions

**What/Why:**  
Functions are first-class in R, enabling modular code, pipelines, and custom analysis steps. Anonymous functions (lambdas) are easy to define inline.

**Example:**

```r
# Function with default argument
greet <- function(name, msg = "Hello") {
  paste(msg, name)
}
print(greet("Data Scientist"))
print(greet("Data Scientist", msg = "Welcome"))

# Anonymous function in apply
nums <- c(2, 4, 8)
doubled <- sapply(nums, function(x) x * 2)
print(doubled)
```

**You should be able to:**  
- Write and call functions with positional, named, and default arguments  
- Use anonymous functions within apply/map calls  
- Return multiple values as lists

---

## 6. Packages & The Tidyverse

**What/Why:**  
R’s package ecosystem is vast. The "tidyverse" (including `dplyr`, `tidyr`, `ggplot2`, `readr`, `purrr`) is foundational for modern data science workflows—enabling readable, chainable "pipelines".

**Example:**

```r
# Install (once) and load tidyverse
# install.packages("tidyverse")
library(tidyverse)

# Data manipulation with dplyr
df <- tibble(name = c("Alice", "Bob"), age = c(25, 30))
df2 <- df %>%
  filter(age > 25) %>%
  mutate(age_group = if_else(age >= 30, "senior", "junior"))
print(df2)

# Visualization with ggplot2
ggplot(df, aes(x = name, y = age)) +
  geom_col() +
  labs(title = "Ages by Name")
```

**You should be able to:**  
- Install and load packages  
- Use `%>%` for pipelines  
- Manipulate data with `dplyr`, visualize with `ggplot2`, import data with `readr`

---

## 7. Reading & Writing Data

**What/Why:**  
Getting data in and out of R is fundamental. R supports CSV, Excel, databases, and more.

**Example:**

```r
# Read CSV
iris <- read.csv("data/iris.csv")
head(iris)

# Write CSV
write.csv(iris, "data/iris_copy.csv", row.names = FALSE)
```

**You should be able to:**  
- Read/write CSV and Excel files  
- Use `readr::read_csv()` for faster, more robust import  
- Explore data with `head()`, `summary()`, and `str()`

---

## 8. Error Handling

**What/Why:**  
Handling errors gracefully is essential for robust scripts and analyses.

**Example:**

```r
tryCatch({
  val <- as.numeric("not a number")
}, warning = function(w) {
  print(paste("Warning:", w))
}, error = function(e) {
  print(paste("Error:", e))
}, finally = {
  print("Done attempting conversion.")
})
```

**You should be able to:**  
- Use `tryCatch` for error and warning handling  
- Write robust, user-friendly error messages  
- Debug with `traceback()` and `browser()`

---

## 9. Environments & Scoping

**What/Why:**  
Understanding environments helps you reason about variable scope and function behavior—especially in complex scripts or packages.

**Example:**

```r
x <- 5
f <- function() {
  x <- 10
  print(x)
}
f()        # prints 10
print(x)   # prints 5 (outer x unchanged)
```

**You should be able to:**  
- Recognize variable scope (global vs. local)  
- Use environments for advanced programming

---

# R Ecosystem for Data Science

These packages will be covered in depth in later modules, but here are the essentials:

## Tidyverse

A collection of packages (`dplyr`, `ggplot2`, `tidyr`, `readr`, `purrr`, `tibble`) for data wrangling, visualization, and analysis in a unified, consistent style. The `%>%` "pipe" operator enables readable, stepwise transformations.

## Data.table

High-performance data manipulation for large datasets. Syntax is concise and blazing fast for filtering, aggregation, and joins.

## Base R

Functions like `aggregate()`, `by()`, `reshape()`, and base plotting are always available.

## Caret

Unified interface for machine learning models and workflows (similar to scikit-learn in Python).

## Shiny

For building interactive web apps with R.

---

# Key Takeaways

- R is built for data science—vectors, data frames, and the tidyverse are your core tools.
- Master vectorized operations, apply-family functions, and data frame manipulation.
- The R ecosystem offers robust packages for import, wrangling, visualization, and modeling.
- Robust workflow includes using scripts, RMarkdown/Quarto, error handling, and reproducible environments (e.g., renv, packrat).

---

### Exercise

"""
Create a numeric vector containing the numbers 1 to 10, then use vectorized operations to generate a vector of their squares and print the result.
"""

```r
# Write your solution below
numbers <- 1:10
squares <- 
# TODO: your code here
print(squares)
```