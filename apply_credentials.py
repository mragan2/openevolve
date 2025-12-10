import os

# --- PATHS ---
BASE_DIR = os.getcwd()
CONFIG_PATH = os.path.join(BASE_DIR, "examples", "hubble_tension", "config.yaml")

# --- UPDATED CONFIGURATION ---
NEW_CONFIG = """# OpenEvolve Configuration (Hubble Tension - Custom Endpoint)
max_iterations: 100
checkpoint_interval: 10
log_level: "INFO"

llm:
  # The Cloud Model you requested
  primary_model: "qwen3-coder:480b-cloud"
  primary_model_weight: 1.0
  
  # --- YOUR INSERTED CREDENTIALS ---
  api_base: "http://localhost:11434/v1"
  api_key: "${OPENAI_API_KEY}"
  # ---------------------------------
  
  temperature: 0.85
  top_p: 0.95
  max_tokens: 8192
  timeout: 600

prompt:
  system_message: |
    Jesteś wybitnym fizykiem. Musisz rozwiązać "Hubble Tension" (Napięcie Hubble'a).
    
    OBECNY MODEL:
    Twój model Massive Graviton działa jak Stała Kosmologiczna (a_factor = 1.0).
    To pasuje do H0_Late (73 km/s/Mpc), ale jest sprzeczne z H0_Early (67 km/s/Mpc).

    ZADANIE:
    Zmodyfikuj funkcję `H_mg_phenomenological`, aby Ciemna Energia była DYNAMICZNA.
    Gęstość Ciemnej Energii musi się zmieniać w czasie (nie może być stała!).
    Spróbuj skalowania typu: `a_factor = a ** epsilon` (gdzie epsilon jest małe, np. 0.1 lub -0.1).
    
    CELE:
    1. Idealne dopasowanie do H0_LATE (73 km/s/Mpc) dzisiaj (a=1).
    2. Zachowanie gęstości próżni (~0.7 * rho_crit).
    3. Zmienna gęstość w czasie (Dynamical Dark Energy).

    Używaj stałych. Kod w Pythonie.

  num_top_programs: 4
  use_template_stochasticity: true

database:
  population_size: 50
  archive_size: 20
  num_islands: 3
  elite_selection_ratio: 0.2
  exploitation_ratio: 0.6

evaluator:
  timeout: 60
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.75]
  parallel_evaluations: 1
  use_llm_feedback: false

evolution_settings:
  diff_based_evolution: false
  allow_full_rewrites: true
  max_code_length: 10000
"""

def main():
    print(f"--- 🔑 UPDATING HUBBLE CONFIG CREDENTIALS 🔑 ---")
    
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: Config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(NEW_CONFIG)

    print("✅ Config updated successfully.")
    print("   Model: qwen3-coder:480b-cloud")
    print("   Endpoint: http://localhost:11434/v1")
    print("   Key: aa24...9w0")
    print("\n👉 Run 'run_hubble.bat' to start.")

if __name__ == "__main__":
    main()