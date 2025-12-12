import json
import os
import shutil

# SETTINGS
TRACE_FILE = "logs/phase1_trace.jsonl"
OUTPUT_FILENAME = "seed_for_phase2.py" # Phase 2 will look for this

def extract_best():
    if not os.path.exists(TRACE_FILE):
        print(f"❌ Error: {TRACE_FILE} not found. Run Phase 1 first!")
        return False

    best_score = -9999
    best_code = None
    
    print(f"🔍 Scanning {TRACE_FILE}...")
    
    with open(TRACE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Look for combined_score or score
                metrics = entry.get('metrics', {})
                score = metrics.get('combined_score', metrics.get('score', 0))
                
                if score > best_score and 'code' in entry:
                    best_score = score
                    best_code = entry['code']
            except:
                continue

    if best_code:
        with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
            f.write(best_code)
        print(f"✅ FOUND BEST CANDIDATE (Score: {best_score})")
        print(f"📄 Saved as: {OUTPUT_FILENAME}")
        return True
    else:
        print("❌ No valid code found in trace.")
        return False

if __name__ == "__main__":
    extract_best()