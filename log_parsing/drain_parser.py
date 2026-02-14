import os
import re
import pandas as pd

def parse_logs(input_path, output_path):
    print("[INFO] Parsing logs...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    event_templates = {}
    event_ids = []
    template_id = 0

    with open(input_path, 'r') as f:
        for line in f:
            log_message = re.sub(r'\d+', '<*>', line.strip())

            if log_message not in event_templates:
                event_templates[log_message] = template_id
                template_id += 1

            event_ids.append(event_templates[log_message])

    df = pd.DataFrame({"event_id": event_ids})
    df.to_csv(output_path, index=False)

    print(f"[INFO] Parsed {len(event_ids)} log lines.")