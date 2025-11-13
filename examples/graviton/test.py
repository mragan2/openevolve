from pathlib import Path
import importlib.util

# Resolve paths relative to this file
HERE = Path(__file__).resolve().parent
evaluator_path = HERE / "evaluator.py"
program_path = HERE / "initial_program.py"

print(f"Evaluator path: {evaluator_path}")
print(f"Program path:   {program_path}")

# Load evaluator module from file
spec = importlib.util.spec_from_file_location("eval_mod", evaluator_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not create spec for {evaluator_path}")

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Call the evaluate() function from evaluator.py
if not hasattr(mod, "evaluate"):
    raise AttributeError("evaluator.py has no function named 'evaluate'")

result = mod.evaluate(str(program_path))
print("Evaluation result:")
print(result)
