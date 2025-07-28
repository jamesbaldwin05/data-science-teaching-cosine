import streamlit as st
import os
import re
from pathlib import Path
from utils.code_runner import run_code
from utils.progress import load_progress, save_progress

MODULES_DIR = Path(__file__).resolve().parent / "modules"
DATA_DIR = Path(__file__).resolve().parent / "data"
PROGRESS_PATH = Path(__file__).resolve().parent / ".progress.json"

ORDER = ["beginner", "intermediate", "advanced"]
CATEGORY_NAMES = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced"
}

def list_categories_and_modules():
    categories = []
    category_to_modules = {}
    for cname in ORDER:
        category_dir = MODULES_DIR / cname
        if category_dir.is_dir():
            categories.append(cname)
            md_files = sorted([f for f in category_dir.glob("*.md") if f.is_file()])
            category_to_modules[cname] = md_files
    return categories, category_to_modules

def parse_markdown_sections(md_text):
    sections = {}
    cur_section = None
    cur_lines = []
    for line in md_text.splitlines():
        heading = re.match(r"^###?\s*(\w+)", line)
        if heading:
            if cur_section:
                sections[cur_section] = "\n".join(cur_lines).strip()
            cur_section = heading.group(1).lower()
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_section:
        sections[cur_section] = "\n".join(cur_lines).strip()
    return sections

def parse_quiz_block(quiz_md):
    # Split into question blocks by **Q
    blocks = re.split(r'(?=\*\*Q\d+:)', quiz_md)
    result = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # **Qn:** ...\n (question line)
        qid_match = re.match(r"\*\*Q(\d+):\*\*\s*(.+)", block)
        if not qid_match:
            continue
        qid = qid_match.group(1)
        question = qid_match.group(2).strip()
        # Find choices
        choices = re.findall(r"^-\s*([A-D])\)\s*(.+)$", block, re.MULTILINE)
        # Find answer (multi-choice: letter; text: string)
        ans_match = re.search(r"\*\*A:\*\*\s*(.+)", block)
        answer = ans_match.group(1).strip() if ans_match else ""
        is_text = False
        if choices:
            # Multi-choice: answer is e.g. "B"
            pass
        else:
            is_text = True
        result.append({
            "qid": qid,
            "question": question,
            "choices": choices,
            "answer": answer,
            "is_text": is_text
        })
    return result

def extract_codeblock(md, section_name):
    pattern = rf"###\s*{section_name}.*?```python(.*?)```"
    match = re.search(pattern, md, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_exercise_instructions(md):
    pattern = r"###\s*Exercise\s*[\r\n]+\"\"\"(.*?)\"\"\""
    match = re.search(pattern, md, re.DOTALL)
    return match.group(1).strip() if match else ""

def get_default_selection(categories, category_to_modules):
    for cname in ORDER:
        if category_to_modules.get(cname):
            first_mod = category_to_modules[cname][0]
            return f"{cname}/{first_mod.name}"
    return ""

def main():
    st.set_page_config(page_title="Data Science for Developers", layout="wide")
    st.title("🧑‍💻 Data Science for Developers")

    categories, category_to_modules = list_categories_and_modules()
    progress = load_progress(PROGRESS_PATH)

    # Default selection, persisted in session_state
    all_paths = []
    cat_mods = {}
    for cat in categories:
        cat_mods[cat] = []
        for mod in category_to_modules[cat]:
            path_str = f"{cat}/{mod.name}"
            cat_mods[cat].append(path_str)
            all_paths.append(path_str)

    if "selected_module" not in st.session_state:
        st.session_state["selected_module"] = get_default_selection(categories, category_to_modules)
    selected_path = st.session_state["selected_module"]

    # --- Sidebar tree UI ---
    st.sidebar.title("Curriculum")
    # --- Improved sidebar module selection logic ---
    # Determine selected category for radio
    if selected_path and "/" in selected_path:
        selected_category = selected_path.split("/")[0]
    else:
        selected_category = categories[0]

    # --- Improved sidebar module selection logic ---
    # Determine selected category for radio
    if selected_path and "/" in selected_path:
        selected_category = selected_path.split("/")[0]
    else:
        selected_category = categories[0]

    chosen = None
    for cat in categories:
        # Determine if all modules in this category are completed
        cat_complete = True
        for mod in category_to_modules[cat]:
            mod_id = f"{cat[0]}_{mod.stem[:2]}"
            if not progress.get(mod_id, {}).get("quiz_completed"):
                cat_complete = False
                break
        exp_label = f"{CATEGORY_NAMES[cat]} {'✅' if cat_complete else ''}"
        with st.sidebar.expander(exp_label, expanded=(cat==selected_category)):
            cur_paths = cat_mods[cat]
            for i, mod in enumerate(category_to_modules[cat]):
                mod_id = f"{cat[0]}_{mod.stem[:2]}"
                completed = progress.get(mod_id, {}).get("quiz_completed", False)
                # Always show numeric prefix (i+1:02d) and derive title from stem after first underscore
                # This guarantees the numeric prefix is present and readable
                label_text = f"{i+1:02d}. {' '.join(mod.stem.split('_')[1:]).title()}"
                label = label_text + (" ✅" if completed else "")
                is_selected = (cur_paths[i] == selected_path)
                button_key = f"select_{cat}_{i}"
                btn_label = label
                if st.button(btn_label, key=button_key):
                    st.session_state["selected_module"] = cur_paths[i]
                    st.rerun()
                if is_selected:
                    chosen = (cat, mod)
    # Fallback if not chosen
    if not chosen:
        cat = categories[0]
        chosen = (cat, category_to_modules[cat][0])
        st.session_state["selected_module"] = f"{cat}/{category_to_modules[cat][0].name}"

    selected_category, selected_mod = chosen
    mod_id = f"{selected_category[0]}_{selected_mod.stem[:2]}"

    # Read lesson
    md_text = selected_mod.read_text(encoding="utf-8")
    sections = parse_markdown_sections(md_text)

    # --- Main lesson display ---
    st.markdown(md_text.split("### Example")[0])

    # Example code
    example_code = extract_codeblock(md_text, "Example")
    if example_code:
        with st.expander("Show Example Code", expanded=True):
            st.code(example_code, language="python")
            if st.button("Run Example", key=f"run_ex_{mod_id}"):
                output, error = run_code(example_code)
                st.text_area("Output", output + (f"\n[Error]: {error}" if error else ""), height=150)

    # Exercise
    if "exercise" in sections:
        st.subheader("Exercise")
        instructions = extract_exercise_instructions(md_text)
        if instructions:
            st.markdown(f"> {instructions}")
        exercise_code = extract_codeblock(md_text, "Exercise")
        exercise_key = f"exercise_{mod_id}"
        user_code = st.text_area("Edit & Run Your Solution", exercise_code, height=200, key=exercise_key)
        if st.button("Run Exercise", key=f"run_exercise_{mod_id}"):
            output, error = run_code(user_code)
            st.text_area("Exercise Output", output + (f"\n[Error]: {error}" if error else ""), height=150)
            # Update progress
            mod_prog = progress.get(mod_id, {})
            mod_prog["exercise_runs"] = mod_prog.get("exercise_runs", 0) + 1
            progress[mod_id] = mod_prog
            save_progress(PROGRESS_PATH, progress)
            st.success("Exercise run recorded!")

    # Quiz (multi-question)
    if "quiz" in sections:
        st.subheader("Quiz")
        quiz_md = sections["quiz"]
        quiz_questions = parse_quiz_block(quiz_md)
        quiz_key = f"quiz_{mod_id}"
        # Load user answers from session_state, or make new
        if quiz_key not in st.session_state:
            st.session_state[quiz_key] = {}
        user_answers = st.session_state[quiz_key]
        # Show all questions
        for q in quiz_questions:
            qid = q["qid"]
            field_key = f"{quiz_key}_{qid}"
            if q["choices"]:
                options = [f"{c[0]}) {c[1]}" for c in q["choices"]]
                answer_val = user_answers.get(qid, "")
                idx = None
                for i, opt in enumerate(q["choices"]):
                    if answer_val == opt[0]:
                        idx = i
                choice = st.radio(q["question"], options, key=field_key, index=idx if idx is not None else 0)
                # Save letter (e.g. 'A') on change
                letter = choice.split(")")[0]
                user_answers[qid] = letter
            else:
                answer_val = user_answers.get(qid, "")
                resp = st.text_input(q["question"], value=answer_val, key=field_key)
                user_answers[qid] = resp
        # Quiz submit
        if st.button("Submit Quiz", key=f"submit_quiz_{mod_id}"):
            all_correct = True
            results = []
            for q in quiz_questions:
                qid = q["qid"]
                ua = user_answers.get(qid, "")
                if not q["is_text"]:
                    # Multi-choice
                    correct = ua.upper().strip() == q["answer"].upper().strip()
                else:
                    correct = ua.strip().lower() == q["answer"].strip().lower()
                results.append(correct)
                if not correct:
                    all_correct = False
            if all_correct:
                st.success("✅ All answers correct!")
                mod_prog = progress.get(mod_id, {})
                mod_prog["quiz_completed"] = True
                progress[mod_id] = mod_prog
                save_progress(PROGRESS_PATH, progress)
            else:
                for i, correct in enumerate(results):
                    if not correct:
                        st.error(f"❌ Q{i+1}: Incorrect.")
                st.info("Please review the incorrect answers and try again.")
        if progress.get(mod_id, {}).get("quiz_completed"):
            st.info("You have completed this quiz.")

    # Progress summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Progress")
    total_modules = sum(len(ms) for ms in category_to_modules.values())
    completed = sum(1 for k, v in progress.items() if v.get("quiz_completed"))
    st.sidebar.markdown(f"**Quizzes Completed:** {completed} / {total_modules}")
    total_ex = sum(v.get("exercise_runs", 0) for v in progress.values())
    st.sidebar.markdown(f"**Exercises Run:** {total_ex}")

if __name__ == "__main__":
    main()