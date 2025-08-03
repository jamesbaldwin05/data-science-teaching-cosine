import streamlit as st
import os
import re
import sys
from pathlib import Path
from streamlit.components.v1 import html as st_html

def scroll_to_bottom():
    """Scroll browser viewport to page bottom (smooth, retried for Streamlit rerender timing)."""
    st_html(
        """
        <script>
          function scrollBottom(){
            window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
          }
          scrollBottom();
          const times=[100,400,800];
          times.forEach(t=>setTimeout(scrollBottom,t));
        </script>
        """,
        height=0,
    )
from utils.code_runner import run_code
from utils.auth import (
    USERS_PATH, load_users, save_users, hash_password,
    password_valid, verify_credentials, register_user,
    get_user_progress, save_user_progress, ensure_user_exists
)

DEV_MODE = '--dev' in sys.argv
USE_ACE_EDITOR = True  # Feature flag for Ace editor, default ON

def handle_auth():
    """Streamlit UI for login/register/logout. Sets st.session_state['logged_in'] and ['username']."""
    if DEV_MODE:
        st.session_state['logged_in'] = True
        st.session_state['username'] = 'dev'
        return
    if st.session_state.get('logged_in'):
        # Sidebar logout button (not in dev mode)
        with st.sidebar:
            if st.button("Logout", key="logout_btn", help="Log out of your account"):
                st.session_state.pop('logged_in', None)
                st.session_state.pop('username', None)
                st.rerun()
        return

    # Login/Register UI
    tabs = st.tabs(["Login", "Register"])
    login_tab, register_tab = tabs

    with login_tab:
        login_user = st.text_input("Username", key="login_user")
        login_pw = st.text_input("Password", type="password", key="login_pw")
        login_btn = st.button("Login", key="login_btn")
        if login_btn:
            if not login_user or not login_pw:
                st.error("Please enter both username and password.")
            elif verify_credentials(login_user, login_pw):
                st.session_state['logged_in'] = True
                st.session_state['username'] = login_user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        reg_user = st.text_input("Username", key="reg_user")
        reg_pw = st.text_input("Password", type="password", key="reg_pw")
        reg_pw2 = st.text_input("Confirm Password", type="password", key="reg_pw2")
        reg_btn = st.button("Register", key="reg_btn")
        if reg_btn:
            if not reg_user or not reg_pw or not reg_pw2:
                st.error("Please fill in all fields.")
            elif reg_pw != reg_pw2:
                st.error("Passwords do not match.")
            else:
                success, msg = register_user(reg_user, reg_pw)
                if success:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = reg_user
                    st.success("Registration successful! You are now logged in.")
                    st.rerun()
                else:
                    st.error(msg)

    # At the end, if not logged in, halt app execution
    if not st.session_state.get('logged_in'):
        st.stop()

MODULES_DIR = Path(__file__).resolve().parent / "modules"
DATA_DIR = Path(__file__).resolve().parent / "data"


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
    pattern = rf"###\s*{section_name}.*?```(python|r)(.*?)```"
    match = re.search(pattern, md, re.DOTALL | re.IGNORECASE)
    if match:
        lang = match.group(1).strip().lower()
        code = match.group(2).strip()
        return code, lang
    return "", ""

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
    handle_auth()  # Require login/register before showing rest of UI
    st.title("🧑‍💻 Data Science for Developers")
    # Top-of-page one-shot flash message (cleared after display)
    top_flash = st.session_state.pop('top_flash', None)
    if top_flash:
        kind, msg = top_flash
        if kind == 'success':
            st.success(msg)
        elif kind == 'error':
            st.error(msg)
        else:
            st.info(msg)
    # (Removed global flash display: flash messages now shown inline in Exercise section)

    categories, category_to_modules = list_categories_and_modules()

    username = st.session_state.get('username', 'dev')
    if DEV_MODE:
        progress = st.session_state.setdefault('dev_progress', {})
        def persist():
            st.session_state['dev_progress'] = progress
    else:
        ensure_user_exists(username)
        progress = get_user_progress(username)
        def persist():
            save_user_progress(username, progress)

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
    module_counter = 1  # Global module numbering across all categories
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
                completed = progress.get(mod_id, {}).get("exercise_completed") or progress.get(mod_id, {}).get("quiz_completed")
                # Global sequential numbering
                label_text = f"{module_counter:02d} {' '.join(mod.stem.split('_')[1:]).title()}"
                label = label_text + (" ✅" if completed else "")
                is_selected = (cur_paths[i] == selected_path)
                button_key = f"select_{cat}_{i}"
                btn_label = label
                if st.button(btn_label, key=button_key):
                    st.session_state["selected_module"] = cur_paths[i]
                    st.rerun()
                if is_selected:
                    chosen = (cat, mod)
                module_counter += 1
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
    # --- (A) Inline code block runner for Python ---
    import traceback
    if selected_mod.stem == "01_python":
        import matplotlib
        import matplotlib.pyplot as plt
        # Inline parse and render: markdown up to ### Exercise, with code block runners
        code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        exercise_idx = md_text.find("### Exercise")
        if exercise_idx >= 0:
            before_exercise = md_text[:exercise_idx]
        else:
            before_exercise = md_text

        def run_snippet(code, key):
            import streamlit as st
            import io, contextlib, traceback
            import matplotlib.pyplot as plt
            import warnings
            import types

            def get_fallback_vars(gdict):
                # Enumerate non-dunder, non-callable, non-module, user-defined globals
                lines = []
                for k, v in gdict.items():
                    if k.startswith("__") and k.endswith("__"):
                        continue
                    if callable(v):
                        continue
                    if isinstance(v, types.ModuleType):
                        continue
                    try:
                        val_repr = repr(v)
                    except Exception:
                        val_repr = "<unrepresentable>"
                    lines.append(f"{k} = {val_repr}")
                return "\n".join(lines)

            display_code = code
            st.code(display_code, language="python")
            if st.button("Run", key=key):
                with st.spinner("Running..."):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    globals_dict = {}
                    try:
                        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                            exec(code, globals_dict)
                    except Exception:
                        st.error(traceback.format_exc())
                        return
                    # Display matplotlib figure if present
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.close("all")
                    out = stdout.getvalue()
                    err = stderr.getvalue()
                    # Remove non-problematic Matplotlib Agg warning
                    err = "".join([line for line in err.splitlines(keepends=True)
                                   if "FigureCanvasAgg is non-interactive" not in line])
                    # Fallback: if nothing to show, display variable state
                    if (not out.strip()) and (not err.strip()):
                        fallback = get_fallback_vars(globals_dict)
                        if fallback.strip():
                            st.text_area("Output", fallback, height=150, key=f"out_{key}")
                    elif out or err:
                        st.text_area("Output", out + err, height=150, key=f"out_{key}")

        last_pos = 0
        runner_idx = 0
        for match in code_block_pattern.finditer(before_exercise):
            pre_md = before_exercise[last_pos:match.start()]
            if pre_md.strip():
                st.markdown(pre_md)
            code = match.group(1).strip()
            code_lines = code.lstrip().splitlines()
            # B) "no-run": remove line & display code, no button
            if code_lines and code_lines[0].strip().startswith("# no-run"):
                display_code = "\n".join(code_lines[1:]).lstrip() if len(code_lines) > 1 else ""
                if display_code.strip():
                    st.code(display_code, language="python")
                else:
                    st.code("# no-run", language="python")
            else:
                run_snippet(code, key=f"run_snip_{mod_id}_{runner_idx}")
                runner_idx += 1
            last_pos = match.end()
        # Output remaining markdown up to Exercise
        post_md = before_exercise[last_pos:]
        if post_md.strip():
            # Add a blank line to avoid heading merging
            st.markdown(post_md + "\n")
        # Do NOT output after_exercise markdown (no duplication)

    elif selected_mod.stem in ("02_r", "03_r"):
        # Inline parse and render: markdown up to ### Exercise, with code block runners for R
        code_block_pattern = re.compile(r"```r(.*?)```", re.DOTALL | re.IGNORECASE)
        exercise_idx = md_text.find("### Exercise")
        if exercise_idx >= 0:
            before_exercise = md_text[:exercise_idx]
        else:
            before_exercise = md_text

        def run_r_snippet(code, key):
            import streamlit as st
            display_code = code
            st.code(display_code, language="r")
            if st.button("Run", key=key):
                with st.spinner("Running..."):
                    # Use provided code runner utility
                    output, error = run_code(code, lang='r')
                    text_out = (output or "") + (("\n" + error) if error else "")
                    st.text_area("Output", text_out, height=150, key=f"out_{key}")

        last_pos = 0
        runner_idx = 0
        for match in code_block_pattern.finditer(before_exercise):
            pre_md = before_exercise[last_pos:match.start()]
            if pre_md.strip():
                st.markdown(pre_md)
            code = match.group(1).strip()
            code_lines = code.lstrip().splitlines()
            # "no-run": remove line & display code, no button
            if code_lines and code_lines[0].strip().startswith("# no-run"):
                display_code = "\n".join(code_lines[1:]).lstrip() if len(code_lines) > 1 else ""
                if display_code.strip():
                    st.code(display_code, language="r")
                else:
                    st.code("# no-run", language="r")
            else:
                run_r_snippet(code, key=f"run_snip_{mod_id}_{runner_idx}")
                runner_idx += 1
            last_pos = match.end()
        # Output remaining markdown up to Exercise
        post_md = before_exercise[last_pos:]
        if post_md.strip():
            st.markdown(post_md + "\n")
        # Do NOT output after_exercise markdown (no duplication)

    else:
        # --- Fallback: normal markdown render for non-python-overview lessons ---
        st.markdown(md_text.split("### Example")[0])

        # Example code (legacy block, only for non-overview lessons)
        example_code, example_lang = extract_codeblock(md_text, "Example")
        if example_code:
            with st.expander("Show Example Code", expanded=True):
                st.code(example_code, language=example_lang or "python")
                if st.button("Run Example", key=f"run_ex_{mod_id}"):
                    output, error = run_code(example_code, lang=example_lang or "python")
                    st.text_area("Output", output + (f"\n[Error]: {error}" if error else ""), height=150)

    # Exercise
    if "exercise" in sections:
        st.subheader("Exercise")
        instructions = extract_exercise_instructions(md_text)
        if instructions:
            st.markdown(f"> {instructions}")
        exercise_code, exercise_lang = extract_codeblock(md_text, "Exercise")
        exercise_key = f"exercise_{mod_id}"
        if exercise_code is None:
            exercise_code = ""
        # Use ACE editor for Python if enabled; otherwise always use text_area
        if (exercise_lang or "").lower() == "python":
            try:
                ace_val = st_ace(
                    value=exercise_code,
                    language="python",
                    theme="solarized_light",
                    key=f"{exercise_key}_ace",
                    min_lines=12,
                    max_lines=40,
                    height=300,
                    font_size=16,
                    tab_size=4,
                    show_gutter=True,
                    show_print_margin=False,
                    wrap=True,
                    readonly=False,
                    annotations=None,
                    placeholder=None,
                    auto_update=True,
                )
                if ace_val is None:
                    editor = st.text_area(
                        "Your solution:",
                        value=exercise_code,
                        height=300,
                        key=f"{exercise_key}_ta",
                        help="Write your solution here",
                    )
                else:
                    editor = ace_val
            except Exception:
                editor = st.text_area(
                    "Your solution:",
                    value=exercise_code,
                    height=300,
                    key=f"{exercise_key}_ta",
                    help="Write your solution here",
                )
        user_code = editor if editor is not None else exercise_code

        # --- Inline flash message placeholder: appears just under code editor/run area ---
        # We use a container here so the flash message (success/error/info) is always shown in-context,
        # even after reruns, and does not jump to the top of the page.
        flash_container = st.empty()

        if st.button("Run Exercise", key=f"run_exercise_{mod_id}"):
            # Always capture latest code from editor at button press
            user_code = editor if editor is not None else exercise_code
            # Custom logic for 01_python -- check for correct `squares`.
            if selected_mod.stem == "01_python":
                import io, contextlib, traceback
                import types

                stdout, stderr = io.StringIO(), io.StringIO()
                globals_dict = {}
                exception = None
                try:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exec(user_code, globals_dict)
                except Exception:
                    exception = traceback.format_exc()

                squares_expected = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
                has_squares = "squares" in globals_dict
                squares_val = globals_dict["squares"] if has_squares else None
                squares_correct = has_squares and squares_val == squares_expected

                stdout_val = stdout.getvalue()
                err_val = stderr.getvalue()
                # Remove non-problematic Matplotlib Agg warning
                err_val = "".join([line for line in err_val.splitlines(keepends=True)
                                   if "FigureCanvasAgg is non-interactive" not in line])

                printed_correct = squares_correct and str(squares_val) in stdout_val

                # Feedback logic
                if exception:
                    st.error("❌ Your code raised an exception:\n\n" + exception)
                    scroll_to_bottom()
                elif squares_correct and printed_correct:
                    # Immediate feedback, then persist and rerun
                    flash_container.success("✅ Correct! Great job generating and printing the squares.")
                    scroll_to_bottom()
                    st.session_state['flash'] = ('success', '✅ Correct! Great job generating and printing the squares.')
                    st.session_state['top_flash'] = ('success', '✅ Correct! You have completed this module – move on to the next one.')
                    mod_prog = progress.get(mod_id, {})
                    mod_prog["exercise_completed"] = True
                    progress[mod_id] = mod_prog
                    persist()
                    st.rerun()
                elif squares_correct and not printed_correct:
                    st.error("⚠️ You created the correct list but didn't print it. Please add `print(squares)`.")
                    scroll_to_bottom()
                else:
                    msg = "❌ Incorrect – make sure `squares` contains the squares of 1-10."
                    if has_squares:
                        msg += f"\n\nYour `squares`: `{repr(squares_val)}`"
                    else:
                        msg += "\n\nYou did not define the variable `squares`."
                    st.error(msg)
                    scroll_to_bottom()

                # Show captured output (stdout+stderr) for debugging
                if stdout_val or err_val:
                    st.text_area("Exercise Output", stdout_val + err_val, height=150)

                # Record attempt regardless of result
                mod_prog = progress.get(mod_id, {})
                mod_prog["exercise_runs"] = mod_prog.get("exercise_runs", 0) + 1
                progress[mod_id] = mod_prog
                # Replaced broken save_progress(PROGRESS_PATH, progress) with persist()
                # (persist() saves user progress for both dev & normal mode)
                persist()
            else:
                output, error = run_code(user_code, lang=exercise_lang or "python")
                st.text_area("Exercise Output", output + (f"\n[Error]: {error}" if error else ""), height=150)
                # Update progress
                mod_prog = progress.get(mod_id, {})
                mod_prog["exercise_runs"] = mod_prog.get("exercise_runs", 0) + 1
                # If no error, mark exercise as completed
                if not error:
                    mod_prog["exercise_completed"] = True
                    # Immediate feedback so the user gets instant confirmation before rerun
                    flash_container.success("✅ Correct! Exercise run successful.")
                    scroll_to_bottom()
                    # Set flash message to survive rerun on success
                    st.session_state['flash'] = ('success', '✅ Correct! Exercise run successful.')
                    st.session_state['top_flash'] = ('success', '✅ Correct! You have completed this module – move on to the next one.')
                    progress[mod_id] = mod_prog
                    persist()
                    st.rerun()
                else:
                    progress[mod_id] = mod_prog
                    persist()
                    st.success("Exercise run recorded!")
                    scroll_to_bottom()

        # --- Render flash message just under exercise area (always displays in-place) ---
        # This must be outside the Run Exercise button logic so that after rerun, the
        # message is shown promptly in the right context.
        flash = st.session_state.pop('flash', None)
        if flash:
            kind, msg = flash
            if kind == 'success':
                flash_container.success(msg)
                scroll_to_bottom()
            elif kind == 'error':
                flash_container.error(msg)
                scroll_to_bottom()
            else:
                flash_container.info(msg)
                scroll_to_bottom()

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
                st.session_state['top_flash'] = ('success', '✅ Correct! You have completed this module – move on to the next one.')
                mod_prog = progress.get(mod_id, {})
                mod_prog["quiz_completed"] = True
                progress[mod_id] = mod_prog
                persist()
                st.rerun()
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
    exercises_completed = sum(1 for k, v in progress.items() if v.get("exercise_completed"))
    quizzes_completed = sum(1 for k, v in progress.items() if v.get("quiz_completed"))
    st.sidebar.markdown(f"**Exercises Completed:** {exercises_completed} / {total_modules}")

if __name__ == "__main__":
    main()