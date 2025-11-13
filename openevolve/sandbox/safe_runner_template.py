#!/usr/bin/env python3
import sys
import os
import importlib.util
import traceback

def main():
    # Accept program path from argv or environment variable
    if len(sys.argv) > 1 and sys.argv[1].strip():
        program_path = sys.argv[1]
    else:
        program_path = os.environ.get("PROGRAM_PATH")

    if not program_path:
        print("Usage: python safe_runner_template.py <path-to-program.py>")
        print("Or set PROGRAM_PATH environment variable.")
        return 2

    program_path = os.path.abspath(program_path)
    results_path = os.path.splitext(__file__)[0] + ".results"

    # Ensure program directory is on sys.path so the program can import siblings
    sys.path.insert(0, os.path.dirname(program_path))

    print(f"Program path: {program_path}")
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    program = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(program)
    except Exception:
        tb = traceback.format_exc()
        # write traceback to results file
        with open(results_path, "wb") as f:
            f.write(tb.encode("utf-8", errors="replace"))
        print(f"Error saved to {results_path!r}", file=sys.stderr)
        return 1
    else:
        with open(results_path, "wb") as f:
            f.write(b"OK\n")
        print(f"Results saved to {results_path!r}")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())