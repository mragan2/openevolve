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

# --- FINE-TUNE CONFIGURATION ---
# Based on your requested Phase 1 params, but optimized for tuning.
FINETUNE_CONFIG = """
# OpenEvolve Configuration (Phase 2: Fine-Tuning)
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  primary_model: "qwen3-coder:480b-cloud"
  primary_model_weight: 1.0
  api_base: "http://localhost:11434/v1"
  api_key: "aa249496fa974637a67ebe8f05be1e21.bfs5CdlZ_ocSK0O__Guty9w0"
  
  # LOWER TEMPERATURE FOR PRECISION
  temperature: 0.4  
  top_p: 0.90
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś precyzyjnym fizykiem. Masz działający model "Phase Transition Dark Energy".
    
    OBECNY STAN:
    Model jest świetny (Score ~0.98). Używa przejścia sigmoidalnego.
    
    ZADANIE:
    Dostrój parametry (`transition_midpoint`, `transition_width`, `epsilon`), aby uzyskać PERFEKCYJNY wynik.
    
    CELE:
    1. H_today musi wynosić DOKŁADNIE 73.000 km/s/Mpc.
    2. Gęstość musi wynosić 0.700 * rho_crit.
    
    UWAGA: Nie zmieniaj struktury funkcji! Tylko liczby.

  num_top_programs: 3
  use_template_stochasticity: false

database:
  population_size: 60      # Your requested size
  archive_size: 25
  num_islands: 4           # Your requested islands
  elite_selection_ratio: 0.3
  exploitation_ratio: 0.8  # Increased to focus on best solution

evaluator:
  timeout: 60
  cascade_evaluation: true
  cascade_thresholds: [0.8, 0.9] # Stricter thresholds
  parallel_evaluations: 4
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: true   # CHANGED TO TRUE: Critical for fine-tuning numbers
  allow_full_rewrites: false   # CHANGED TO FALSE: Protects the algorithm structure
  max_code_length: 10000
"""

def main():
    print("--- 🎯 SETTING UP FINE-TUNING RUN 🎯 ---")

    # 1. Promote the Best Program
    if os.path.exists(BEST_PROG):
        print(f"1. Promoting Winner: {BEST_PROG}")
        shutil.copy(BEST_PROG, TARGET_SEED)
    else:
        print("❌ Error: No best program found to fine-tune!")
        return

    # 2. Write the Config
    print("2. Writing Fine-Tune Config...")
    with open(TARGET_CONFIG, "w", encoding="utf-8") as f:
        f.write(FINETUNE_CONFIG)

    # 3. Clean Old Output (To start fresh from high score)
    print("3. Cleaning old database for fresh run...")
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR, ignore_errors=True)

    print("\n✅ READY. The AI will now polish your Phase Transition model.")
    print("   Run 'run_hubble.bat' to start.")

if __name__ == "__main__":
    main()