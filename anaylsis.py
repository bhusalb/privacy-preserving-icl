import json
import os
import sys
import numpy as np

def load_jsonl_file(filename):
    """
    Load a .jsonl file and return a list of JSON objects.

    Args:
        filename (str): Path to the .jsonl file.

    Returns:
        list: A list of JSON objects (dictionaries).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' does not exist.")

    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        data = load_jsonl_file(filepath)
        
        sorted_data = sorted(data, key=lambda x: x['accuracy_with_dp_cali'], reverse=True)
        top3 = sorted_data[:5]
        # top3 = sorted_data
        keys = ['accuracy_without_dp', 'accuracy_without_dp_cali', 'accuracy_with_dp', 'accuracy_with_dp_cali']
        stats = {}

        for key in keys:
            values = [entry[key] for entry in top3]
            stats[key] = {
                'mean': float(np.mean(values)) * 100,  # convert to percentage
                'std': float(np.std(values, ddof=1)) * 100 # sample std
            }

        print("\nTop 3 Summary Statistics:")
        for metric, values in stats.items():
            print(f"{metric}: mean = {values['mean']:.4f}, std = {values['std']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
