import os
import shutil
import glob

# --- PATHS ---
BASE_DIR = os.getcwd()
SOLVE_DIR = os.path.join(BASE_DIR, "examples", "solve")
BEST_PROG_DIR = os.path.join(SOLVE_DIR, "openevolve_output", "best")
TARGET_FILE = os.path.join(SOLVE_DIR, "initial_program.py")
CONFIG_FILE = os.path.join(SOLVE_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(SOLVE_DIR, "openevolve_output")
CACHE_DIR = os.path.join(SOLVE_DIR, "__pycache__")

# --- STAGE 2 CONFIGURATION (Mixed: Your Params + Physics Context) ---
STAGE_2_CONFIG = """
# OpenEvolve Configuration (Stage 2: Optimization)
# Based on your "Part 2" parameters but tailored for Graviton Physics

max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

# LLM CONFIGURATION (Keeping Local Qwen as requested)
llm:
  primary_model: "qwen2.5-coder:7b"
  primary_model_weight: 0.8
  secondary_model: "mistral:7b"
  secondary_model_weight: 0.2
  api_base: "http://localhost:11434/v1"
  api_key: "ollama"
  temperature: 0.7
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

# PROMPT CONFIGURATION (Polish Physics Context)
prompt:
  system_message: |
    Jesteś wybitnym fizykiem teoretycznym. 
    Twoim zadaniem jest udoskonalenie teorii Masywnej Grawitacji (Massive Gravity).
    
    Startujesz z bardzo dobrego rozwiązania (Stage 1 Best). 
    Twoim celem jest optymalizacja i uproszczenie kodu przy zachowaniu idealnych wyników.

    CELE:
    1. H_mg_phenomenological: ~0.7 * H0^2.
    2. rho_quantum: ~0.7 * rho_crit.

    Używaj stałych (c, hbar, M_G_REF).
    Kod musi być poprawnym Pythonem.

  num_top_programs: 4
  use_template_stochasticity: true

# DATABASE CONFIGURATION (Stage 2 Aggressive Settings)
database:
  population_size: 70   # Increased from 30 -> 70
  archive_size: 30      # Increased archive
  num_islands: 5        # Increased diversity (5 islands)
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.6 # Lowered to encourage exploration

# EVALUATOR CONFIGURATION
evaluator:
  timeout: 90           # Increased timeout for complex math
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.8] # Stricter thresholds
  parallel_evaluations: 4
  use_llm_feedback: false

# EVOLUTION SETTINGS
evolution_settings:
  diff_based_evolution: false
  allow_full_rewrites: true
  max_code_length: 100000
"""

def main():
    print("--- 🚀 INITIATING STAGE 2 (PROMOTION) 🚀 ---")

    # 1. Locate the Best Program from Stage 1
    best_file = os.path.join(BEST_PROG_DIR, "best_program.py")
    
    if not os.path.exists(best_file):
        print(f"❌ Error: Could not find best program at {best_file}")
        print("   Did the previous run finish correctly?")
        return

    print(f"1. Promoting Winner: {best_file}")
    
    # Read the winner
    with open(best_file, "r", encoding="utf-8") as f:
        winner_code = f.read()

    # Write it as the NEW seed
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(winner_code)
    
    print(f"   ✅ {TARGET_FILE} updated with Stage 1 Winner.")

    # 2. Update Configuration
    print("2. Writing Stage 2 Config (Pop 70, Islands 5)...")
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(STAGE_2_CONFIG)

    # 3. Clean Old Database (So we start fresh from the new high baseline)
    print("3. Cleaning old database for fresh Stage 2 run...")
    if os.path.exists(OUTPUT_DIR):
        try: shutil.rmtree(OUTPUT_DIR)
        except: pass
    if os.path.exists(CACHE_DIR):
        try: shutil.rmtree(CACHE_DIR)
        except: pass

    print("\n✅ STAGE 2 READY.")
    print("   The AI will now start with your perfect solution and try to optimize it further.")
    print("   Run 'run_experiment.bat' to launch.")

if __name__ == "__main__":
    main()