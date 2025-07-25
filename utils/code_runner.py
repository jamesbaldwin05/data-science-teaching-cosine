import sys
import io
import traceback

def run_code(code: str):
    """Safely execute code and capture stdout/stderr, returning (out, error)"""
    stdout = io.StringIO()
    stderr = io.StringIO()
    globals_dict = {}
    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        exec(code, globals_dict)
        out = stdout.getvalue()
        err = stderr.getvalue()
        if err:
            return out, err
        return out, ""
    except Exception as e:
        tb = traceback.format_exc()
        return stdout.getvalue(), tb
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr