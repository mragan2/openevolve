diff --git a/examples/graviton/evaluator.py b/examples/graviton/evaluator.py
index 90616f6f8afcc1c8d7e6ede276288e2376f951f6..34ecfa99fa7b4935c661124dd898be4a0c9a3cb6 100644
--- a/examples/graviton/evaluator.py
+++ b/examples/graviton/evaluator.py
@@ -1,76 +1,210 @@
-"""
-Evaluator for massive graviton / dark energy models in OpenEvolve.
+"""Evaluator for massive graviton / dark energy models in OpenEvolve.
+
+Historically the graviton task asked models to emit the *entire* scaffolded
+program, but in practice most models only emit the EVOLVE block.  When that
+happens the old evaluator attempted to import the partial file directly which
+obviously failed (none of the required helper functions existed), so every
+candidate received a zero score and the search could not make progress.
+
+To fix this we now mirror the more robust quantum task evaluator:
+
+* Extract the user supplied EVOLVE block (or fall back to importing the module
+  directly if it already contains the full scaffold).
+* Splice the evolved block into ``initial_program.py`` which is kept on disk as
+  the canonical scaffold.
+* Execute the stitched program in isolation and expose the resulting namespace
+  to the downstream scoring logic.
+
+This keeps backwards compatibility—candidates who still output the entire file
+continue to work—while also allowing the cheaper “block only” workflow that the
+other tasks use.  The rest of the evaluator logic is unchanged.
 
 Expected functions in the candidate module:
   - graviton_mass_from_lambda(lambda_g_m: float) -> float
   - yukawa_potential(r: float, M: float, lambda_g_m: float) -> float
   - gw_group_velocity(omega: float, m_g: float) -> float
   - lambda_eff_from_mg(m_g: float) -> float
   - H_mg_phenomenological(a: float, m_g: float, H0: float) -> float
   - build_massive_gravity_model(...) -> dict
 
 Optionally:
   - run_sanity_checks() -> dict
 
 CRITICAL (per OpenEvolve README):
   - evaluate(program_path) must return a dictionary, not EvaluationResult.
   - The dict MUST include 'combined_score' as the primary metric.
   - On failure: combined_score = 0.0 and an 'error' key is recommended.
 """
 
 import importlib.util
 import math
+import tempfile
 import time
 import uuid
+from pathlib import Path
+from types import SimpleNamespace
 from typing import Any, Dict
 
+import runpy
+
 
 # Reference constants (must match initial_program constants)
 C_LIGHT = 299_792_458.0
 HBAR = 1.054_571_817e-34
 G_NEWTON = 6.674_30e-11
 
 # Reference graviton Compton wavelength and mass
 LAMBDA_G_REF_METERS = 4.39e26  # ≈ 4.64 gly
 M_G_EXPECTED = HBAR / (C_LIGHT * LAMBDA_G_REF_METERS)
 
 # Observational scales
 LAMBDA_EFF_REF = 1.0e-52   # m^-2, order of observed cosmological constant
 OMEGA_MG_REF = 0.7         # present-day dark-energy fraction
 H0_REF = 2.2e-18           # s^-1
 
 
-def _load_candidate_module(program_path: str):
-    """Dynamically import the candidate program as a module."""
+def _extract_evolve_block(source: str) -> str:
+    """Return the code between EVOLVE markers in *source*.
+
+    This mirrors the behaviour used by other examples.  We require the block to
+    exist and be non-empty.  The helper raises ``ValueError`` when the markers
+    are missing so that the evaluator can fall back to importing the module
+    directly (useful when a candidate emits the entire scaffold).
+    """
+
+    start_tag = "# EVOLVE-BLOCK-START"
+    end_tag = "# EVOLVE-BLOCK-END"
+    lines = source.splitlines()
+
+    start_idx = None
+    end_idx = None
+    for idx, line in enumerate(lines):
+        if line.strip() == start_tag:
+            start_idx = idx
+            continue
+        if line.strip() == end_tag:
+            end_idx = idx
+            break
+
+    if start_idx is None or end_idx is None or end_idx <= start_idx:
+        raise ValueError("Missing EVOLVE block markers")
+
+    block = "\n".join(lines[start_idx + 1 : end_idx]).strip()
+    if not block:
+        raise ValueError("EVOLVE block is empty")
+    return block
+
+
+def _sanitize_user_block(block: str) -> str:
+    """Strip directives that would be invalid inside the EVOLVE block."""
+
+    sanitized_lines = []
+    for line in block.splitlines():
+        stripped = line.strip()
+        if stripped.startswith("from __future__ import"):
+            continue
+        sanitized_lines.append(line)
+
+    result = "\n".join(sanitized_lines).strip()
+    if not result:
+        raise ValueError("EVOLVE block became empty after sanitization")
+    return result
+
+
+def _build_locked_program(scaffold_source: str, user_block: str) -> str:
+    """Splice *user_block* into the scaffold between the EVOLVE markers."""
+
+    lines = scaffold_source.splitlines()
+    out_lines = []
+    in_block = False
+
+    start_tag = "# EVOLVE-BLOCK-START"
+    end_tag = "# EVOLVE-BLOCK-END"
+
+    for line in lines:
+        stripped = line.strip()
+        if stripped == start_tag:
+            out_lines.append(line)
+            out_lines.append(user_block)
+            in_block = True
+            continue
+        if stripped == end_tag and in_block:
+            in_block = False
+            out_lines.append(line)
+            continue
+        if not in_block:
+            out_lines.append(line)
+
+    return "\n".join(out_lines)
+
+
+def _execute_stitched_program(full_source: str) -> SimpleNamespace:
+    """Execute *full_source* in isolation and return it as a namespace."""
+
+    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
+        tmp.write(full_source)
+        tmp_path = Path(tmp.name)
+
+    try:
+        globals_dict = runpy.run_path(str(tmp_path), run_name="__candidate__")
+        return SimpleNamespace(**globals_dict)
+    finally:
+        try:
+            tmp_path.unlink()
+        except OSError:
+            pass
+
+
+def _import_full_module(program_path: str) -> SimpleNamespace:
+    """Import the candidate file directly when it already contains the scaffold."""
+
     module_name = f"candidate_massive_{uuid.uuid4().hex}"
     spec = importlib.util.spec_from_file_location(module_name, program_path)
     if spec is None or spec.loader is None:
         raise ImportError(f"Could not create spec for {program_path}")
     module = importlib.util.module_from_spec(spec)
     spec.loader.exec_module(module)  # type: ignore[attr-defined]
-    return module
+    return SimpleNamespace(**module.__dict__)
+
+
+def _load_candidate_module(program_path: str) -> SimpleNamespace:
+    """Load a candidate file, stitching it into the scaffold when necessary."""
+
+    program_path = Path(program_path)
+    source = program_path.read_text(encoding="utf-8")
+
+    scaffold_path = Path(__file__).resolve().parent / "initial_program.py"
+    try:
+        user_block = _extract_evolve_block(source)
+        sanitized = _sanitize_user_block(user_block)
+        scaffold_source = scaffold_path.read_text(encoding="utf-8")
+        full_source = _build_locked_program(scaffold_source, sanitized)
+        return _execute_stitched_program(full_source)
+    except ValueError:
+        # Candidate probably emitted the full module already; fall back to direct import.
+        return _import_full_module(str(program_path))
 
 
 def _score_from_relative_error(rel_err: float, scale: float = 1.0) -> float:
     """
     Convert a relative error into a [0, 1] score.
 
     rel_err: |x - x_ref| / |x_ref|
     scale: tolerance scale; rel_err ~ scale => score ~ 0.5.
     """
     rel = max(0.0, rel_err / max(scale, 1e-30))
     return 1.0 / (1.0 + rel)
 
 
 def _score_bounded_ratio(val: float, target: float) -> float:
     """
     Score how close val is to target > 0, in log space.
     Score is 1.0 when val ~ target and falls off by orders of magnitude.
     """
     if val <= 0.0 or target <= 0.0:
         return 0.0
     log_ratio = abs(math.log10(val / target))
     return 1.0 / (1.0 + log_ratio)
 
 
 def _clamp01(x: float) -> float:
