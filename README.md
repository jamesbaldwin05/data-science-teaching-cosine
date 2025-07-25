# Data Science for Developers – Streamlit Curriculum

**An interactive, beginner-friendly data science course for experienced programmers – all offline, no external APIs, just Python and open data!**

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

3. **Run the app**  
   ```bash
   streamlit run app.py
   ```

4. **Navigate modules using the sidebar**  
   - Each module includes a concise lesson, runnable code example, interactive exercise (with code you can edit/run), and a quiz.
   - Your progress is saved locally in `.progress.json`.

---

## 📚 Curriculum Structure

- `app.py` — Main Streamlit app
- `modules/` — Markdown lesson files (one per module, editable/extendable)
- `data/` — Included datasets (offline, e.g., iris.csv, titanic.csv)
- `utils/` — Helper functions for safe code execution and progress (see below)
- `assets/` — (Optional) Screenshots/resources

#### Add/Modify Lessons

1. Add a markdown file to `modules/` (see format below).
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

## 💡 Customizing & Contributing

- To add new lessons, just drop a markdown file in `/modules` (follow the template).
- PRs and suggestions welcome!