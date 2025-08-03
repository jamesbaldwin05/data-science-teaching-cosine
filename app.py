import streamlit as st
import os

# ... other necessary imports and setup ...

# EXERCISE EDITOR BLOCK
if (exercise_lang or "").lower() == "python":
            if USE_ACE_EDITOR:
                try:
                    from streamlit_ace import st_ace
                    ace_val = st_ace(
                        value=exercise_code,
                        language="python",
                        theme="solarized_light",
                        key=f"{exercise_key}_ace",
                        height=200,
                        min_lines=8,
                        max_lines=30,
                        font_size=16,
                    )
                    # Fallback if Ace fails to render (returns None)
                    if ace_val is None:
                        editor = st.text_area("Edit & Run Your Solution", exercise_code, height=200, key=f"{exercise_key}_ta")
                    else:
                        editor = ace_val
                except Exception:
                    editor = st.text_area("Edit & Run Your Solution", exercise_code, height=200, key=f"{exercise_key}_ta")
            else:
                editor = st.text_area("Edit & Run Your Solution", exercise_code, height=200, key=f"{exercise_key}_ta")
        else:
            editor = st.text_area("Edit & Run Your Solution", exercise_code, height=200, key=f"{exercise_key}_ta")
    try:
        ace_val = st_ace(
            value=exercise_code,
            language="python",
            key=f"{exercise_key}_ace",
            # ... other st_ace arguments ...
        )
        if ace_val is None:
            ace_val = st.text_area(
                "Write your code here",
                value=exercise_code,
                height=300,
                key=f"{exercise_key}_ta",
            )
    except Exception:
        ace_val = st.text_area(
            "Write your code here",
            value=exercise_code,
            height=300,
            key=f"{exercise_key}_ta",
        )
else:
    ace_val = st.text_area(
        "Write your code here",
        value=exercise_code,
        height=300,
        key=f"{exercise_key}_ta",
    )

# ... rest of the app.py logic ...