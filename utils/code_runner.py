import sys
import io
import traceback

def run_code(code: str):
    """Safely execute code and capture stdout/stderr, returning (out, error)
    If code produces no stdout and no error, output all non-dunder globals as fallback.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    globals_dict = {}
    try:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout, stderr
        exec(code, globals_dict)
        out = stdout.getvalue()
        err = stderr.getvalue()
        # Enhanced stderr handling: only treat as error if Traceback or Error (case-insensitive)
        if err:
            if ("traceback" not in err.lower()) and ("error" not in err.lower()):
                # Not a "real" error: treat as output
                out = out + err
                err = ""
            else:
                return out, err
        if not out.strip():  # No output, no error
            # List all non-dunder globals, skip imports/functions/classes
            lines = []
            for k, v in globals_dict.items():
                if not k.startswith("__") and not callable(v) and not isinstance(v, type(sys)):
                    try:
                        val = repr(v)
                    except Exception:
                        val = "<unprintable>"
                    lines.append(f"{k} = {val}")
            if lines:
                return "\n".join(lines), ""
        return out, ""
    except Exception as e:
        tb = traceback.format_exc()
        return stdout.getvalue(), tb
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr