import os
import shutil
import datetime

# --- CONFIG ---
# We look for your major wins
SOURCE_DIRS = {
    "Project_1_Dark_Energy": "examples/solve",
    "Project_2_Hubble_Tension": "examples/hubble_tension",
    "Project_3_Galaxy_Rotation": "examples/galaxy_rotation",  # NEW: galaxy rotation project
}

# Where to save (Your Desktop)
USER_HOME = os.path.expanduser("~")
DEST_ROOT = os.path.join(USER_HOME, "Desktop", "Massive_Graviton_Discovery_Artifacts")

def preserve():
    print(f"--- 💾 PRESERVING RESEARCH ARTIFACTS TO DESKTOP 💾 ---")
    
    if os.path.exists(DEST_ROOT):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dest = f"{DEST_ROOT}_{timestamp}"
    else:
        final_dest = DEST_ROOT

    os.makedirs(final_dest)
    print(f"Target: {final_dest}")

    for project_name, rel_path in SOURCE_DIRS.items():
        full_src = os.path.abspath(rel_path)
        target_dir = os.path.join(final_dest, project_name)
        
        if not os.path.exists(full_src):
            print(f"⚠️ Warning: Could not find {project_name} at {full_src}")
            continue

        print(f"\n📦 Archiving {project_name}...")
        os.makedirs(target_dir, exist_ok=True)

        # 1. The Best Code
        best_code = os.path.join(full_src, "openevolve_output", "best", "best_program.py")
        if os.path.exists(best_code):
            shutil.copy(best_code, os.path.join(target_dir, "final_physics_model.py"))
            print(f"   - Saved final model code")

        # 2. The Configs (Your IP)
        config = os.path.join(full_src, "config.yaml")
        if os.path.exists(config):
            shutil.copy(config, os.path.join(target_dir, "experiment_config.yaml"))
            print(f"   - Saved experiment configuration")
            
        # 3. The Evaluator (Your Logic)
        evaluator = os.path.join(full_src, "evaluator.py")
        if os.path.exists(evaluator):
            shutil.copy(evaluator, os.path.join(target_dir, "physics_evaluator.py"))
            print(f"   - Saved evaluator logic")

        # 4. Any Plots (The Proof)
        # Look for png files in the repo root that match this project family
        for file in os.listdir(os.getcwd()):
            if (
                file.endswith(".png") and
                (
                    "hubble" in file.lower() or
                    "galaxy" in file.lower()   # NEW: capture galaxy rotation plots
                )
            ):
                shutil.copy(file, os.path.join(target_dir, file))
                print(f"   - Saved plot: {file}")

    print(f"\n✅ SUCCESS. Your data is safe in folder: {final_dest}")
    print("   You can now safely delete the working directories or reinstall the software.")

if __name__ == "__main__":
    preserve()
