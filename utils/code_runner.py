import sys
import io
import traceback

def run_code(code: str, lang: str = 'python'):
    """
    Execute code and capture stdout/stderr, returning (out, error).
    For lang='python', executes as before.
    For lang='r', tries rpy2 then subprocess Rscript as fallback.
    """
    if lang is None:
        lang = 'python'
    if lang.lower() == 'python':
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
    elif lang.lower() == 'r':
        import shutil
        import subprocess
        import tempfile
        import os
        import io
        import contextlib

        # --- 1. Try Rscript first ---
        rscript_path = shutil.which("Rscript") or shutil.which("Rscript.exe")
        rscript_error = ""
        if rscript_path is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix='.R', delete=False) as tmp_file:
                tmp_file.write(code)
                tmp_file_path = tmp_file.name
            try:
                proc = subprocess.run(
                    [rscript_path, tmp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if proc.returncode == 0:
                    return proc.stdout, proc.stderr
                else:
                    rscript_error = proc.stderr or f"Rscript exited with code {proc.returncode}"
            except Exception as e:
                rscript_error = f"Error running Rscript: {str(e)}"
            finally:
                try:
                    os.remove(tmp_file_path)
                except Exception:
                    pass
        else:
            rscript_error = "Rscript not found in PATH."

        # --- 2. Try rpy2 fallback, wrap import and execution in stdout/stderr capture ---
        try:
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                import rpy2
                import rpy2.robjects as robjects
                import rpy2.rinterface_lib.callbacks

                # Setup rpy2 output/error capturing
                @contextlib.contextmanager
                def capture_r_output():
                    old_writeconsole = rpy2.rinterface_lib.callbacks.consolewrite_print
                    old_writeconsole_warn = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
                    rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: output_buffer.write(x)
                    rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: error_buffer.write(x)
                    try:
                        yield
                    finally:
                        rpy2.rinterface_lib.callbacks.consolewrite_print = old_writeconsole
                        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = old_writeconsole_warn

                with capture_r_output():
                    robjects.r(code)
            out = output_buffer.getvalue()
            err = error_buffer.getvalue()
            return out, err
        except Exception as e_rpy2:
            comb_err = f"Rscript failed: {rscript_error}; rpy2 failed: {e_rpy2}"
            return "", comb_err
    else:
        return "", f"Language '{lang}' is not supported."