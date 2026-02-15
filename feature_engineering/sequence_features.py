import os
import pandas as pd
import numpy as np

def build_sequences(input_path,output_path,window_size):
    print("[INFO] Building sequences...")
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df=pd.read_csv(input_path)
    events=df['event_id'].values
    sequences=[]
    for i in range(len(events)-window_size):
        sequences.append(events[i:i+window_size])
    sequences=np.array(sequences)
    np.save(output_path,sequences)
    print(f"[INFO] Generated {len(sequences)} sequences.")