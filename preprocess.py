import os
import yaml
import sys
import pandas as pd

# Get the base directory (root of the project)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the correct path to params.yaml
params_path = os.path.join(base_dir, "params.yaml")

# Load YAML file
try:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)['preprocess']
    print("Loaded parameters:", params)
except Exception as e:
    print(f"Error loading {params_path}: {e}")
    sys.exit(1)

def preprocess(input_path, output_path):
    print(f"Reading input file from: {input_path}")

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        sys.exit(1)

    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    print("Executing preprocessing function...")
    input_file = os.path.join(base_dir, params['input'])
    output_file = os.path.join(base_dir, params['output'])
    
    preprocess(input_file, output_file)
    print(f"Reading input file from: {input_file}")
    print(f"Saving output file to: {output_file}")
