import os
import shutil

# --- PATHS ---
BASE_DIR = os.getcwd()
HUBBLE_DIR = os.path.join(BASE_DIR, "examples", "hubble_tension")
BEST_PROG = os.path.join(HUBBLE_DIR, "openevolve_output", "best", "best_program.py")
TARGET_SEED = os.path.join(HUBBLE_DIR, "initial_program.py")
TARGET_CONFIG = os.path.join(HUBBLE_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(HUBBLE_DIR, "openevolve_output")
CACHE_DIR = os.path.join(HUBBLE_DIR, "__pycache__")

# --- CREDENTIALS ---
API_BASE = "http://localhost:11434/v1"
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = "qwen3-coder:480b-cloud"

# --- PHASE 3 CONFIGURATION: THE SNIPER ---
HYPERTUNE_CONFIG = f"""
# OpenEvolve Configuration (Phase 3: Hyper-Tuning)
max_iterations: 50  # Short run, just for numbers
checkpoint_interval: 5
log_level: "INFO"

llm:
  primary_model: "{MODEL_NAME}"
  primary_model_weight: 1.0
  api_base: "{API_BASE}"
  api_key: "{API_KEY}"
  
  # FREEZE MODE
  temperature: 0.2  # Extremely low to prevent logic changes
  top_p: 0.85
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś numerycznym solverem. Twoim JEDYNYM celem jest precyzja.
    
    OBECNY WYNIK: 0.9861.
    BRAKUJE: Idealnego dopasowania do H0_LATE = 73.000.
    
    ZADANIE:
    Zmieniaj TYLKO cyfry w parametrach:
    - `transition_midpoint` (np. 0.555 -> 0.55X)
    - `epsilon` (np. -0.085 -> -0.08X)
    
    Nie zmieniaj ani jednej linii kodu logicznego. Tylko liczby.

  num_top_programs: 2
  use_template_stochasticity: false

database:
  population_size: 40      # Focused population
  archive_size: 10
  num_islands: 2           # Less diversity needed
  elite_selection_ratio: 0.4
  exploitation_ratio: 0.9  # 90% exploitation of the best code

evaluator:
  timeout: 60
  cascade_evaluation: true
  cascade_thresholds: [0.9, 0.95] # Only accept excellent programs
  parallel_evaluations: 4
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: true   # MUST be true for number tweaking
  allow_full_rewrites: false   # ABSOLUTELY FORBIDDEN
  max_code_length: 10000
"""

def main():
    print("--- 🎯 INITIATING PHASE 3: HYPER-TUNING 🎯 ---")

    # 1. Promote the 0.9861 Winner
    if os.path.exists(BEST_PROG):
        print(f"1. Promoting Phase 2 Winner: {BEST_PROG}")
        shutil.copy(BEST_PROG, TARGET_SEED)
    else:
        print("❌ Error: No best program found from Phase 2!")
        return

    # 2. Write the Sniper Config
    print("2. Writing Hyper-Tune Config...")
    with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
        f.write(HYPERTUNE_CONFIG)

    # 3. Clean Output
    print("3. Wiping database for final run...")
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR, ignore_errors=True)

    print("\n✅ READY. The AI is locked on target.")
    print("   Run 'run_hubble.bat' to get your 1.0000.")

if __name__ == "__main__":
    main()