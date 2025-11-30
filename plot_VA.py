import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = './dataset/zho_restaurant_train_alltasks.jsonl'
VALENCE_OUTPUT_FILE = 'valence_distribution_en.png' # Changed output name
AROUSAL_OUTPUT_FILE = 'arousal_distribution_en.png' # Changed output name
BIN_SIZE = 0.2
MIN_VAL = 1.0
MAX_VAL = 9.0

def extract_va_scores(file_name: str) -> pd.DataFrame:
    """Reads the JSONL file and extracts Valence and Arousal scores from all Quadruplets."""
    
    if not os.path.exists(file_name):
        print(f"Error: Input file '{file_name}' not found. Please check the file path.")
        return pd.DataFrame()

    va_scores = []
    
    print(f"Starting to read file: {file_name}")
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Iterate through every quadruplet
                for quad in data.get('Quadruplet', []):
                    va_str = quad.get('VA')
                    if va_str and '#' in va_str:
                        # Split and convert to float
                        valence, arousal = map(float, va_str.split('#'))
                        va_scores.append({'Valence': valence, 'Arousal': arousal})
            
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
            
    print(f"Successfully extracted {len(va_scores)} VA scores.")
    return pd.DataFrame(va_scores)

def plot_distribution(data_series: pd.Series, title_prefix: str, filename: str, color: str, bins: np.ndarray):
    """Plots the score distribution as a bar chart and saves it as a PNG file."""
    
    # Bin the data using the specified 0.2 intervals
    binned_counts = pd.cut(
        data_series, 
        bins=bins, 
        include_lowest=True, 
        right=True 
    ).value_counts(sort=False)

    # Format X-axis labels (e.g., (4.0-4.2])
    bin_labels = [f"({b.left:.1f}-{b.right:.1f}]" for b in binned_counts.index]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Score_Range': bin_labels, 
        'Count': binned_counts.values
    })
    
    # Filter out bins with zero count
    plot_df = plot_df[plot_df['Count'] > 0]
    
    plt.figure(figsize=(14, 7))
    plt.bar(plot_df['Score_Range'], plot_df['Count'], width=0.8, color=color, edgecolor='black')
    
    # --- English Titles and Labels ---
    plt.title(f'{title_prefix} Score Distribution (Bin Size: {BIN_SIZE})', fontsize=18)
    plt.xlabel('Score Range', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Rotate X-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Display count on top of bars
    for i, count in enumerate(plot_df['Count']):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(filename)
    plt.close()
    
    print(f"Successfully saved chart: {filename}")

def main():
    """Main execution function."""
    
    # 1. Extract data
    df = extract_va_scores(INPUT_FILE)
    
    if df.empty:
        return

    # 2. Define bin edges
    bins = np.arange(MIN_VAL, MAX_VAL + 2 * BIN_SIZE, BIN_SIZE)

    # 3. Plot Valence (V) chart
    plot_distribution(
        df['Valence'], 
        'Valence (Pleasure Score)', # English Title
        VALENCE_OUTPUT_FILE,
        'skyblue',
        bins
    )

    # 4. Plot Arousal (A) chart
    plot_distribution(
        df['Arousal'], 
        'Arousal (Activation Score)', # English Title
        AROUSAL_OUTPUT_FILE,
        'lightcoral',
        bins
    )
    
    print("\nAll English charts have been generated.")

if __name__ == '__main__':
    # Execute the plotting logic
    main()