# run_ardor.py — one-file launcher for the EXE

import os, sys, inspect, runpy, multiprocessing
from pathlib import Path

# --- Where are we running from? (works for onedir & onefile) ---
if getattr(sys, "frozen", False):
    # onefile => sys._MEIPASS is the unpack dir; onedir => folder with the EXE
    BASE = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
else:
    BASE = Path(__file__).resolve().parent  # ...\Praetor\ArdorApp

# Prefer bundled DLLs (fixes Tcl/Tk lookup on Windows)
try:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(BASE))
except Exception:
    pass

# Tcl/Tk resources are bundled under lib\tcl8.6 and lib\tk8.6
os.environ.setdefault("TCL_LIBRARY", str((BASE / "lib" / "tcl8.6").resolve()))
os.environ.setdefault("TK_LIBRARY",  str((BASE / "lib" / "tk8.6").resolve()))

# --- Project root resolution ---
# If a build script wrote ardor_home.txt next to the EXE, honor it.
ah = BASE / "ardor_home.txt"
if ah.exists():
    try:
        PROJECT_ROOT = Path(ah.read_text(encoding="utf-8").strip()).resolve()
    except Exception:
        PROJECT_ROOT = BASE
else:
    # Walk upward to find the repo root (look for Cerebrum or Praetor)
    cur = BASE
    PROJECT_ROOT = BASE
    for _ in range(6):
        if (cur / "Cerebrum").exists() or (cur / "Praetor").exists():
            PROJECT_ROOT = cur
            break
        cur = cur.parent

# Make imports work (both bundled & from source)
for p in (
    BASE,
    PROJECT_ROOT,
    PROJECT_ROOT / "Praetor",
    PROJECT_ROOT / "Praetor" / "ArdorApp",
    PROJECT_ROOT / "Hephaestus",
    PROJECT_ROOT / "Cerebrum",
    PROJECT_ROOT / "Cerebrum" / "Cortex",
    PROJECT_ROOT / "Cerebrum" / "ProjectTokenizer",
):
    try:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    except Exception:
        pass

def rpath(*parts: str) -> Path:
    """Resolve a resource path included via --add-data OR available in the project tree."""
    p = BASE.joinpath(*parts)
    return p if p.exists() else PROJECT_ROOT.joinpath(*parts)

class PATHS:
    ROOT       = PROJECT_ROOT
    DIST_BASE  = BASE
    CEREBRUM   = rpath("Cerebrum")
    CORTEX     = rpath("Cerebrum", "Cortex")
    MODELS     = rpath("Cerebrum", "Models")
    TOKENIZER  = rpath("Cerebrum", "ProjectTokenizer", "ardor_tokenizer")
    REM        = rpath("Cerebrum", "CorticalIntegration", "REM.py")
    TRAIN      = rpath("Cerebrum", "Cortex", "neural_plasticity_training.py")

def _call_if_present(mod, name):
    if hasattr(mod, name) and callable(getattr(mod, name)):
        fn = getattr(mod, name)
        try:
            return fn(PATHS) if len(inspect.signature(fn).parameters) >= 1 else fn()
        except ValueError:
            # Builtins or C funcs can raise on signature; just call with no args
            return fn()

def main():
    # Keep relative paths stable at runtime
    try:
        os.chdir(BASE)
    except Exception:
        pass

    # Try explicit entry points first
    import GUI_Cortex as GC

    out = _call_if_present(GC, "main")
    if out is not None:
        return out

    # If there is an App class, instantiate and run it
    if hasattr(GC, "App"):
        App = GC.App
        try:
            if "paths" in inspect.signature(App.__init__).parameters:
                app = App(paths=PATHS)
            else:
                app = App()
        except Exception:
            app = GC.App()  # best effort
        try:
            return app.mainloop()
        except AttributeError:
            # some apps expose start() instead
            starter = getattr(app, "start", None)
            if callable(starter):
                return starter()

    # Final fallback: execute GUI_Cortex as if run directly (triggers its __main__ block)
    runpy.run_module("GUI_Cortex", run_name="__main__")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
