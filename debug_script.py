import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traceback

# Create directories
print("Creating directories...")
try:
    os.makedirs("plots/additional_analysis", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    print("Directories created successfully")
except Exception as e:
    print(f"Error creating directories: {str(e)}")
    traceback.print_exc()

# Test file writing in results directory
print("\nTesting file writing capabilities...")
try:
    # Create a simple DataFrame
    test_df = pd.DataFrame({
        'imputation_method': ['Mean', 'MICE', 'MissForest'],
        'missing_percentage': [0.25, 0.25, 0.25],
        'RMSE': [0.5, 0.4, 0.3]
    })

    # Try to save it
    test_path = "results/test_output.csv"
    test_df.to_csv(test_path, index=False)

    if os.path.exists(test_path):
        print(f"Successfully wrote test file to {test_path}")
        print(f"File size: {os.path.getsize(test_path)} bytes")
    else:
        print(f"File writing failed - {test_path} not found after writing")
except Exception as e:
    print(f"Error writing test file: {str(e)}")
    traceback.print_exc()

# Test plot creation
print("\nTesting plot creation...")
try:
    plt.figure(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Test Plot')

    test_plot_path = "plots/additional_analysis/test_plot.png"
    plt.savefig(test_plot_path)
    plt.close()

    if os.path.exists(test_plot_path):
        print(f"Successfully saved plot to {test_plot_path}")
        print(f"File size: {os.path.getsize(test_plot_path)} bytes")
    else:
        print(f"Plot saving failed - {test_plot_path} not found after saving")
except Exception as e:
    print(f"Error creating test plot: {str(e)}")
    traceback.print_exc()

# Check current working directory and absolute paths
print("\nDirectory information:")
print(f"Current working directory: {os.getcwd()}")
print(f"Absolute path to results dir: {os.path.abspath('results')}")
print(f"Absolute path to plots dir: {os.path.abspath('plots/additional_analysis')}")

# List contents of current directory
print("\nContents of current directory:")
for item in os.listdir('.'):
    print(f"- {item}")

# Try to load combined_results_summary.csv if it exists anywhere
print("\nSearching for combined_results_summary.csv:")
potential_paths = [
    'results/combined_results_summary.csv',
    'combined_results_summary.csv',
    '../results/combined_results_summary.csv',
    'Imputation_MCL/results/combined_results_summary.csv'
]

for path in potential_paths:
    if os.path.exists(path):
        print(f"Found at: {path}")
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns: {', '.join(df.columns)}")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
    else:
        print(f"Not found at: {path}")