#!/usr/bin/env python3
"""
Environment & compatibility checker (solutions + autograder requirements).

Windows-safe update: multiprocessing spawn test now uses a top-level function
so it is picklable on 'spawn' (required by Windows/macOS).

Exit codes:
 0 -> all REQUIRED checks pass
 1 -> some REQUIRED checks fail (suggestions printed)
 2 -> unexpected failure
"""

import sys, traceback

OK = "✅"
FAIL = "❌"
WARN = "⚠️"

REQUIRED_ISSUES = []
SUGGESTIONS = []
WARNINGS = []

def log(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

# --- Top-level child function for multiprocessing 'spawn' ---
def _child_put_42(q):
    q.put(42)

def check_python_core():
    major, minor, micro = sys.version_info[:3]
    print(f"- Python runtime: {major}.{minor}.{micro}")

    # f-strings (3.6+)
    try:
        x = 7
        assert eval("f'x={x}'") == "x=7"
        print(f"{OK} f-strings are supported (PEP 498).")
    except Exception:
        REQUIRED_ISSUES.append("f-strings are NOT supported (need Python ≥ 3.6).")
        SUGGESTIONS.append("Upgrade to Python 3.6+ (3.10+ recommended).")

    # dataclasses (3.7+ built-in; 3.6 backport acceptable)
    try:
        import dataclasses  # noqa: F401
        print(f"{OK} 'dataclasses' module available.")
    except Exception:
        REQUIRED_ISSUES.append("'dataclasses' module is missing.")
        if (major, minor) == (3, 6):
            SUGGESTIONS.append("Install backport: pip install dataclasses (or upgrade to Python 3.7+).")
        else:
            SUGGESTIONS.append("Upgrade to Python 3.7+ where 'dataclasses' is built-in.")

    # typing basics (PEP 484/526)
    try:
        code = (
            "from typing import Optional, Union, List, Dict\n"
            "x: int = 0\n"
            "y: Optional[int] = None\n"
            "z: Union[int, str] = 1\n"
            "a_list: List[int] = [1, 2]\n"
            "a_dict: Dict[str, int] = {'k': 1}\n"
            "def f(a: int) -> int:\n"
            "    return a\n"
        )
        compile(code, "<typing_basics>", "exec")
        print(f"{OK} Typing basics (PEP 484/526) are supported.")
    except SyntaxError:
        REQUIRED_ISSUES.append("Typing annotations (PEP 484/526) are NOT supported by this interpreter.")
        SUGGESTIONS.append("Upgrade to Python 3.7+.")

    # TypedDict REQUIRED at runtime by solution
    try:
        from typing import TypedDict  # type: ignore
        print(f"{OK} typing.TypedDict available (Python ≥ 3.8).")
    except Exception:
        try:
            from typing_extensions import TypedDict  # type: ignore
            print(f"{OK} typing_extensions.TypedDict available (Python 3.7).")
        except Exception:
            REQUIRED_ISSUES.append("TypedDict support is missing (used by the solution at runtime).")
            SUGGESTIONS.append("On Python 3.7: pip install typing_extensions  • OR • upgrade to Python 3.8+.")

    # PEP 604 unions OPTIONAL
    try:
        compile("from typing import Any\nv: Any | None = None\n", "<pep604>", "exec")
        print(f"{OK} PEP 604 union operator ('X | Y') is supported (optional).")
    except SyntaxError:
        WARNINGS.append("PEP 604 unions ('X | Y') not supported (optional). Use Optional[T] on Python < 3.10 if needed.")

def check_packages_and_plotting():
    # numpy (REQUIRED)
    try:
        import numpy as np  # noqa: F401
        print(f"{OK} numpy available.")
    except Exception:
        REQUIRED_ISSUES.append("numpy is not importable.")
        SUGGESTIONS.append("Install with: pip install numpy")

    # matplotlib.pyplot (REQUIRED by grader.py)
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        print(f"{OK} matplotlib.pyplot importable.")
        # Optional: headless backend check
        try:
            import matplotlib as mpl
            mpl.use('Agg', force=True)
            import matplotlib.pyplot as plt2
            fig = plt2.figure()
            plt2.close(fig)
            print(f"{OK} Headless plotting available (Agg backend).")
        except Exception:
            WARNINGS.append("Headless plotting check failed; on servers, ensure a non-interactive backend (Agg).")
    except Exception:
        REQUIRED_ISSUES.append("matplotlib.pyplot is not importable (needed by grader/benchmarks).")
        SUGGESTIONS.append("Install with: pip install matplotlib")

def check_multiprocessing_spawn():
    # Multiprocessing spawn context (REQUIRED by graderUtil timeouts)
    try:
        import multiprocessing as mp
        # Ensure 'spawn' start method (default on Windows/macOS)
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            # Already set; that's fine
            pass
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_child_put_42, args=(q,))
        p.start()
        p.join(5)
        ok = (not p.is_alive()) and (not p.exitcode) and (q.get_nowait() == 42)
        if ok:
            print(f"{OK} multiprocessing 'spawn' context usable.")
        else:
            raise RuntimeError(f"child alive={p.is_alive()} exitcode={p.exitcode}")
    except Exception as e:
        REQUIRED_ISSUES.append("multiprocessing 'spawn' context not usable (autograder timeouts depend on it).")
        SUGGESTIONS.append("On Windows/macOS this should work by default. If running from a notebook/REPL, run as a script with 'python check_env_full_winfix.py'.")
        SUGGESTIONS.append(f"Details: {type(e).__name__}: {e}")

def main():
    log("Python core & typing features")
    check_python_core()

    log("Packages & plotting")
    check_packages_and_plotting()

    log("Multiprocessing spawn")
    check_multiprocessing_spawn()

    log("Summary")
    if WARNINGS:
        print("Notes:")
        for w in WARNINGS:
            print(f"  {WARN} {w}")

    if not REQUIRED_ISSUES:
        print(f"{OK} All REQUIRED checks passed. Environment is suitable.")
        sys.exit(0)
    else:
        print(f"{FAIL} Issues detected:")
        for i, msg in enumerate(REQUIRED_ISSUES, 1):
            print(f"  {i}. {msg}")
        if SUGGESTIONS:
            print("\nSuggestions:")
            for s in SUGGESTIONS:
                print(f"  - {s}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Required by Windows when using multiprocessing
        from multiprocessing import freeze_support
        freeze_support()
        main()
    except Exception:
        print(f"{FAIL} Unexpected failure:\n{traceback.format_exc()}")
        sys.exit(2)
