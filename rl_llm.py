import json
import time
import os
import glob
from transformers import pipeline

print("[RL-LLM] Loading AI Models...")
generator = pipeline("text-generation", model="gpt2")

ACTIONS = ["NO-OP", "ALERT-ADMIN", "ISOLATE-NETWORK", "EMERGENCY-LOCKDOWN"]
RESULTS_PATH = "C:/tmp/sentinel_results/*.json"

def get_latest_fused_score():
    """Reads the most recent JSON file produced by Spark."""
    files = glob.glob(RESULTS_PATH)
    if not files:
        return None
    
    # Get the newest file
    latest_file = max(files, key=os.path.getctime)
    try:
        with open(latest_file, 'r') as f:
            # Spark JSON output can have multiple lines
            for line in f:
                data = json.loads(line)
                return data.get("FUSED_SCORE", 0.0)
    except:
        return None

def generate_summary_with_llm(score, action):
    prompt = f"Security Alert: Fused Threat Score is {score}. Action: {action}. Analysis:"
    result = generator(prompt, max_new_tokens=25, num_return_sequences=1, truncation=True)
    return result[0]['generated_text'].replace(prompt, "").strip()

def process_events():
    print("[SYSTEM] RL + LLM Layer listening to Spark results...")
    last_processed_score = -1

    while True:
        current_score = get_latest_fused_score()
        
        # Only process if we have a new score and it's different from the last one
        if current_score is not None and current_score != last_processed_score:
            action_idx = 3 if current_score > 70 else 2 if current_score > 40 else 1
            action = ACTIONS[action_idx]
            
            summary = generate_summary_with_llm(current_score, action)
            
            event = {
                "timestamp": time.ctime(),
                "fused_score": round(current_score, 2),
                "rl_action": action,
                "llm_summary": summary,
                "assets": ["Network-Gateway", "CCTV-Main"]
            }
            
            with open("business_stream.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
            
            print(f"[REAL-TIME DATA] Score: {current_score:.2f} | Action: {action}")
            last_processed_score = current_score
        
        time.sleep(2) # Poll every 2 seconds

if __name__ == "__main__":
    process_events()