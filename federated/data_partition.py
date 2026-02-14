import numpy as np
import os

def partition_data(input_path, output_dir, num_clients=3):
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(input_path)
    split_data = np.array_split(data, num_clients)

    for i, chunk in enumerate(split_data):
        np.save(f"{output_dir}/client_{i}.npy", chunk)

    print(f"Data split into {num_clients} clients.")