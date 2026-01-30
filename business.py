import json
import os
import time
import pandas as pd
from datetime import datetime

STREAM_FILE = "business_stream.jsonl"
CSV_FILE = "business_analytics.csv"

print("--- Business Analytics Dashboard ---")

processed_lines = 0

while True:
    if os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, "r") as f:
            all_lines = f.readlines()
        
        if len(all_lines) > processed_lines:
            new_lines = all_lines[processed_lines:]
            data = [json.loads(line) for line in new_lines]
            
            df = pd.DataFrame(data)
            
            # Save to CSV
            header = not os.path.exists(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', index=False, header=header)
            
            # Write Markdown Summary
            with open("summary.md", "w") as m:
                m.write(f"# Sentinel Security Executive Summary\n")
                m.write(f"**Last Updated:** {datetime.now()}\n\n")
                m.write(f"| Metric | Value |\n| :--- | :--- |\n")
                m.write(f"| Total Incidents | {len(all_lines)} |\n")
                m.write(f"| Latest Action | {data[-1]['rl_action']} |\n")
                m.write(f"| AI Analysis | {data[-1]['llm_summary']} |\n")

            processed_lines = len(all_lines)
            print(f"[BUSINESS] Updated report with {len(data)} new events.")
            
    time.sleep(5)