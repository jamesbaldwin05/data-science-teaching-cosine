import streamlit as st
import os
import re
from pathlib import Path
from utils.code_runner import run_code
from utils.progress import load_progress, save_progress

MODULES_DIR = Path(__file__).resolve().parent / "modules"
DATA_DIR = Path(__file__).resolve().parent / "data"
PROGRESS_PATH = Path(__file__).resolve().parent / ".progress.json"

CATEGORY_NAMES = {
    "beginner": "Beginner",
    "intermediate": "Intermediate",
    "advanced": "Advanced"
}

def list_categories_and_modules():
    categories = []
    category_to_modules = {}
    for category_dir in sorted(MODULES_DIR.iterdir()):
        if category_dir.is_dir() and category_dir.name in CATEGORY_NAMES:
            categories.append(category_dir.name)
            md_files = sorted([f for f in category_dir.glob("*.md") if f.is_file()])
            category_to_modules[category_dir.name] = md_files
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
    q_match = re.search(r"\*\*Question:\*\*\s*(.+)", quiz_md)
    answer_match = re.search(r"\*\*Answer:\*\*\s*([A-D])", quiz_md)
    choices = re.findall(r"^-\s*([A-D])\)\s*(.+)$", quiz_md, re.MULTILINE)
    question = q_match.group(1).strip() if q_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return question, choices, answer

def extract_codeblock(md, section_name):
    pattern = rf"###\s*{section_name}.*?```python(.*?)```"
    match = re.search(pattern, md, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_exercise_instructions(md):
    pattern = r"###\s*Exercise\s*[\r\n]+\"\"\"(.*?)\"\"\""
    match = re.search(pattern, md, re.DOTALL)
    return match.group(1).strip() if match else ""

def main():
    st.set_page_config(page_title="Data Science for Developers", layout="wide")
    st.title("🧑‍💻 Data Science for Developers")

    categories, category_to_modules = list_categories_and_modules()

    if not categories:
        st.error("No modules found.")
        return

    st.sidebar.title("Curriculum")

    # First sidebar: category
    category_idx = st.sidebar.radio(
        "Section",
        list(enumerate([CATEGORY_NAMES[c] for c in categories])),
        format_func=lambda x: x[1],
        index=0
    )
    selected_category = categories[category_idx[0]]
    modules = category_to_modules[selected_category]
    module_names = [f"{i+1:02d}. {m.stem[3:].replace('_', ' ').title()}" for i, m in enumerate(modules)]

    # Second sidebar: module within category
    module_idx = st.sidebar.radio(
        "Module",
        list(enumerate(module_names)),
        format_func=lambda x: x[1],
        index=0
    )
    selected_mod = modules[module_idx[0]]
    # Progress key is f"{category[0]}_{file.stem[:2]}"
    mod_id = f"{selected_category[0]}_{selected_mod.stem[:2]}"

    # Load progress
    progress = load_progress(PROGRESS_PATH)

    # Read lesson
    md_text = selected_mod.read_text(encoding="utf-8")
    sections = parse_markdown_sections(md_text)

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

    # Quiz
    if "quiz" in sections:
        st.subheader("Quiz")
        q_md = sections["quiz"]
        question, choices, answer = parse_quiz_block(q_md)
        choice_labels = [f"{opt}) {txt}" for opt, txt in choices]
        quiz_key = f"quiz_{mod_id}"
        user_choice = st.radio(question, choice_labels, key=quiz_key)
        if st.button("Submit Quiz", key=f"submit_quiz_{mod_id}"):
            chosen_letter = user_choice.split(")")[0]
            if chosen_letter == answer:
                st.success("✅ Correct!")
                mod_prog = progress.get(mod_id, {})
                mod_prog["quiz_completed"] = True
                progress[mod_id] = mod_prog
                save_progress(PROGRESS_PATH, progress)
            else:
                st.error(f"❌ Incorrect. The correct answer is {answer})")
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