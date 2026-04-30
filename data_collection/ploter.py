import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def plot_initial_distribution(data_dir):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    initial_positions = []

    print(f"Found {len(csv_files)} episode files. Processing...")

    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract the first row as the initial state
            if not df.empty:
                init_x = df.iloc[0]['box_x']
                init_y = df.iloc[0]['box_y']
                initial_positions.append({'box_x': init_x, 'box_y': init_y})
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Convert to DataFrame for plotting
    dist_df = pd.DataFrame(initial_positions)

    # Plotting using Matplotlib
    plt.figure(figsize=(10, 7))
    plt.scatter(dist_df['box_x'], dist_df['box_y'], c='blue', alpha=0.6, edgecolors='w', s=100, label='Starting Position')
    
    # # Optional: Highlight the intended initialization region (-7.0 < x < -6.0 and 1.0 < y < 2.0)
    # plt.axvspan(-7.0, -6.0, color='gray', alpha=0.1, label='Target Init Region')
    # plt.axhspan(1.0, 2.0, color='gray', alpha=0.1)

    plt.title('Distribution of Initial Object Positions ($x, y$)')
    plt.xlabel('Initial Box X Coordinate')
    plt.ylabel('Initial Box Y Coordinate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the figure
    plt.savefig('initial_pose_distribution.png')
    print("Plot saved as 'initial_pose_distribution.png'")
    
    # Export summary data
    dist_df.to_csv('initial_positions_summary.csv', index=False)
    return dist_df

# Usage
initial_data = plot_initial_distribution('success_7Hz')