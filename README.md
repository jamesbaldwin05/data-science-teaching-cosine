# Data Science for Developers – Streamlit Curriculum

**An interactive, beginner-to-advanced data science course for experienced programmers – all offline, no external APIs, just Python and open data!**

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone &lt;this_repo&gt;
   cd &lt;this_repo&gt;
   ```

2. **Install requirements**  
   Python 3.9+ recommended  
   ```bash
   pip install -r requirements.txt
   ```
   *Optional: For some advanced modules, you may want to install extra packages (see below).*

3. **Run the app**  
   ```bash
   streamlit run app.py
   ```

4. **Navigate curriculum in the sidebar**  
   - First choose a section (Beginner, Intermediate, Advanced), then pick a module.
   - Each module includes a lesson, runnable code example, interactive exercise, and a quiz.
   - Your progress is saved locally in `.progress.json`.

---

## 🗂️ Curriculum Outline

**Beginner**
- 01 Python Overview
- 02 R Overview
- 03 Stats & Math Basics
- 04 Data Loading & Visualization
- 05 Pandas Basics
- 06 Data Exploration

**Intermediate**
- 01 NumPy Introduction
- 02 Data Cleaning
- 03 Feature Engineering
- 04 Supervised Learning
- 05 Model Evaluation
- 06 Unsupervised Learning
- 07 Advanced Visualization
- 08 SQL Basics
- 09 Mini Project: Titanic

**Advanced**
- 01 Deep Learning Introduction
- 02 API Data Ingestion
- 03 Bayesian Inference
- 04 Cloud Services for DS
- 05 MLOps & Deployment

---

## 📁 Project Structure

- `app.py` — Main Streamlit app
- `modules/` — Contains three folders: `beginner/`, `intermediate/`, `advanced/`; each with lesson markdowns
- `data/` — Included datasets (offline, e.g., iris.csv, titanic.csv)
- `utils/` — Helper functions for safe code execution and progress
- `assets/` — (Optional) Screenshots/resources

#### Add/Modify Lessons

1. Add a markdown file to the relevant subfolder in `modules/` (see format below).
2. The app will auto-detect and display it in the sidebar.

**Lesson markdown format:**

```
# Title

**Concept** (plain-language explanation)

### Example
```python
# runnable example code
```

### Exercise
"""
Task instructions here.
"""
```python
# skeleton for learner to edit
```

### Quiz
**Question:** Your question here?
- A) Option 1
- B) Option 2
**Answer:** B
```

---

## 🛠️ Utilities

- `utils/code_runner.py` — Safely runs code, captures output/errors
- `utils/progress.py` — Loads/saves local `.progress.json` (quiz/exercise progress)

---

## 🗃️ Datasets

- All required CSVs are in `/data`.
- Example code references data files using relative paths so everything works offline.

---

## ⚡ Advanced/Optional Dependencies

Some advanced modules (Deep Learning, API, Cloud, etc.) use extra libraries.
**These are NOT required for the core curriculum.**
- To enable all modules:  
  ```bash
  pip install torch requests boto3 tensorflow
  ```
- You can run just the beginner/intermediate modules with the default requirements.

---

## 💡 Customizing & Contributing

- To add new lessons, just drop a markdown file in the right subfolder in `/modules` (follow the template).
- PRs and suggestions welcome!