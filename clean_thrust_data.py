"""
Thrust Data Noise Cleaning Program
Removes noise from thrust vs time CSVs using various filtering methods
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import uniform_filter1d
import sys
import os
import matplotlib.pyplot as plt

# ========== CONFIGURE YOUR CSV FILE HERE ==========
INPUT_CSV = "gosha vs goddard - ThrustVsTime_Dirty.csv"
OUTPUT_FOLDER = "cleaned_data"  # All cleaned CSVs and graphs go here
# Other options:
#   INPUT_CSV = "inputs/inputs_vertical_sample-goddard.csv"
#   INPUT_CSV = "inputs/inputs_vertical_sample-gosha.csv"
#   INPUT_CSV = "output_thrust.csv"
# =================================================

def load_data(filepath):
    """Load CSV file and identify thrust column"""
    df = pd.read_csv(filepath)
    
    # Try to find thrust column (case-insensitive)
    thrust_col = None
    time_col = None
    
    for col in df.columns:
        if 'thrust' in col.lower():
            thrust_col = col
        if 'time' in col.lower():
            time_col = col
    
    if thrust_col is None:
        print(f"Available columns: {list(df.columns)}")
        thrust_col = input("Enter thrust column name: ").strip()
    
    if time_col is None:
        time_col = df.columns[0]  # Use first column as time
    
    return df, time_col, thrust_col


def check_data_quality(df, thrust_col):
    """Check for missing or invalid data"""
    thrust = df[thrust_col].values
    
    # Try to convert to numeric
    thrust_numeric = pd.to_numeric(thrust, errors='coerce')
    
    nan_count = np.sum(np.isnan(thrust_numeric))
    valid_count = len(thrust) - nan_count
    
    print(f"\n📊 Data Quality Check:")
    print(f"  Total rows: {len(thrust)}")
    print(f"  Valid thrust values: {valid_count}")
    print(f"  Missing/NaN values: {nan_count}")
    
    if nan_count > 0:
        print(f"  ⚠ Automatically interpolating missing values...")
    
    return thrust_numeric


def moving_average(data, window_size=5):
    """Simple moving average filter"""
    # Handle NaN before filtering
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))
    return uniform_filter1d(data_clean, size=window_size, mode='nearest')


def median_filter(data, kernel_size=5):
    """Median filter - good for spikes"""
    # Handle NaN before filtering
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))
    return medfilt(data_clean, kernel_size=kernel_size)


def savitzky_golay(data, window_length=11, polyorder=3):
    """Savitzky-Golay filter - preserves peaks better"""
    # Handle NaN before filtering
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))
    
    if window_length > len(data_clean):
        window_length = len(data_clean) - 1 if len(data_clean) % 2 == 0 else len(data_clean)
    if window_length < 5:
        window_length = 5
    if window_length % 2 == 0:
        window_length += 1  # Must be odd
    
    return savgol_filter(data_clean, window_length, polyorder)


def exponential_smoothing(data, alpha=0.3):
    """Exponential weighted moving average"""
    # Remove NaN for processing
    clean_data = np.nan_to_num(data, nan=np.nanmean(data))
    
    result = np.zeros_like(clean_data)
    result[0] = clean_data[0]
    for i in range(1, len(clean_data)):
        result[i] = alpha * clean_data[i] + (1 - alpha) * result[i-1]
    return result


def remove_outliers(data, std_threshold=3):
    """Remove points that deviate too much from rolling mean"""
    # Handle NaN first
    data_clean = np.nan_to_num(data, nan=np.nanmean(data))
    
    rolling_mean = pd.Series(data_clean).rolling(window=5, center=True).mean().values
    std_dev = np.nanstd(data_clean)
    mask = np.abs(data_clean - rolling_mean) <= std_threshold * std_dev
    
    # Interpolate removed outliers
    cleaned = data_clean.copy()
    outlier_indices = np.where(~mask)[0]
    for idx in outlier_indices:
        if 0 < idx < len(data_clean) - 1:
            cleaned[idx] = (data_clean[idx-1] + data_clean[idx+1]) / 2
    
    return cleaned


def interactive_menu():
    """Show filter options and get user choice"""
    print("\n=== Thrust Data Cleaning Options ===")
    print("1. Moving Average (smooth, fast)")
    print("2. Median Filter (good for spikes)")
    print("3. Savitzky-Golay (preserves peaks)")
    print("4. Exponential Smoothing (weighted)")
    print("5. Remove Outliers + Smoothing (combined)")
    print("6. Apply All Filters (creates multiple outputs)")
    
    choice = input("\nSelect filter method (1-6): ").strip()
    return choice


def clean_data(df, thrust_col, method='savgol', **kwargs):
    """Apply selected cleaning method"""
    time_col = df.columns[0]  # Assume first column is time
    
    # Convert to float (handles NaN and non-numeric values)
    thrust = pd.to_numeric(df[thrust_col], errors='coerce').values
    
    if method == 'moving_avg':
        window = kwargs.get('window', 5)
        cleaned = moving_average(thrust, window_size=window)
        
    elif method == 'median':
        kernel = kwargs.get('kernel', 5)
        cleaned = median_filter(thrust, kernel_size=kernel)
        
    elif method == 'savgol':
        window = kwargs.get('window', 11)
        poly = kwargs.get('polyorder', 3)
        cleaned = savitzky_golay(thrust, window_length=window, polyorder=poly)
        
    elif method == 'exponential':
        alpha = kwargs.get('alpha', 0.3)
        cleaned = exponential_smoothing(thrust, alpha=alpha)
        
    elif method == 'outliers':
        threshold = kwargs.get('threshold', 3)
        cleaned = remove_outliers(thrust, std_threshold=threshold)
        
    else:
        cleaned = thrust
    
    return cleaned


def create_output_folder(folder_name):
    """Create output folder if it doesn't exist"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"✓ Created output folder: {folder_name}")


def save_cleaned_data(df, cleaned_thrust, thrust_col, output_path, output_folder):
    """Save cleaned data to CSV in output folder"""
    create_output_folder(output_folder)
    
    full_path = os.path.join(output_folder, output_path)
    df_clean = df.copy()
    df_clean[thrust_col] = cleaned_thrust
    df_clean.to_csv(full_path, index=False)
    print(f"✓ Cleaned data saved: {full_path}")
    
    return full_path


def plot_comparison(df, original_thrust, cleaned_thrust, thrust_col, time_col, method_name, output_folder):
    """Create and display a comparison plot"""
    create_output_folder(output_folder)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Try to get numeric time data
    try:
        time = pd.to_numeric(df[time_col].values, errors='coerce')
        if np.all(np.isnan(time)):
            # If all conversion failed, use index
            time = np.arange(len(df))
    except:
        time = np.arange(len(df))
    
    # Plot 1: Original vs Cleaned
    ax1.plot(time, original_thrust, 'r-', alpha=0.5, linewidth=1, label='Original (Noisy)')
    ax1.plot(time, cleaned_thrust, 'b-', linewidth=2, label='Cleaned')
    ax1.set_xlabel('Time / Index', fontsize=12)
    ax1.set_ylabel('Thrust', fontsize=12)
    ax1.set_title(f'Thrust Comparison - {method_name} Filter', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Noise (difference)
    noise = original_thrust - cleaned_thrust
    ax2.fill_between(time, noise, alpha=0.3, color='orange')
    ax2.plot(time, noise, 'orange', linewidth=1)
    ax2.set_xlabel('Time / Index', fontsize=12)
    ax2.set_ylabel('Noise (Original - Cleaned)', fontsize=12)
    ax2.set_title('Removed Noise', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = f"{os.path.splitext(os.path.basename(INPUT_CSV))[0]}_plot_{method_name}.png"
    plot_path = os.path.join(output_folder, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Graph saved: {plot_path}")
    
    # Show popup
    plt.show()
    plt.close()


def main():
    print("╔════════════════════════════════════════╗")
    print("║  Thrust Data Noise Cleaning Utility    ║")
    print("╚════════════════════════════════════════╝")
    
    # Use configured CSV file
    input_file = INPUT_CSV
    
    try:
        df, time_col, thrust_col = load_data(input_file)
        print(f"\n✓ Loaded: {input_file}")
        print(f"  Time column: {time_col}")
        print(f"  Thrust column: {thrust_col}")
        print(f"  Data points: {len(df)}")
        
        # Check data quality
        checked_thrust = check_data_quality(df, thrust_col)
        
        # Replace with interpolated values
        df[thrust_col] = checked_thrust
        
    except FileNotFoundError:
        print(f"✗ File not found: {input_file}")
        print(f"  Make sure the path is correct relative to: {os.getcwd()}")
        return
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return
    
    # Get filter method
    choice = interactive_menu()
    
    methods = {
        '1': ('moving_avg', {'window': 7}),
        '2': ('median', {'kernel': 7}),
        '3': ('savgol', {'window': 15, 'polyorder': 3}),
        '4': ('exponential', {'alpha': 0.4}),
        '5': ('outliers', {'threshold': 3}),
    }
    
    if choice == '6':
        # Apply all methods
        print("\n⏳ Applying all filters...")
        base_name = os.path.basename(input_file).rsplit('.', 1)[0]
        original = df[thrust_col].values
        
        print(f"\n{'='*60}")
        print(f"Original Data - Total Thrust: {np.sum(original):.2f}")
        print(f"{'='*60}\n")
        
        for method_name, params in methods.values():
            cleaned = clean_data(df, thrust_col, method=method_name, **params)
            output = f"{base_name}_cleaned_{method_name}.csv"
            save_cleaned_data(df, cleaned, thrust_col, output, OUTPUT_FOLDER)
            
            # Show statistics for each filter
            print(f"\n{method_name.upper().replace('_', ' ')}:")
            print(f"  Mean: {np.mean(cleaned):.2f}, Std: {np.std(cleaned):.2f}")
            print(f"  Total Thrust: {np.sum(cleaned):.2f}")
            print(f"  Noise reduction: {(1 - np.std(cleaned)/np.std(original))*100:.1f}%")
            
            plot_comparison(df, original, cleaned, thrust_col, time_col, method_name, OUTPUT_FOLDER)
        
        print("\n✓ All filters applied successfully!")
        
    elif choice in methods:
        method_name, params = methods[choice]
        
        # Ask for custom parameters
        if method_name == 'moving_avg':
            window = input("Window size (default 7): ").strip()
            params['window'] = int(window) if window else 7
        
        elif method_name == 'median':
            kernel = input("Kernel size (default 7): ").strip()
            params['kernel'] = int(kernel) if kernel else 7
        
        elif method_name == 'savgol':
            window = input("Window length (default 15): ").strip()
            poly = input("Polynomial order (default 3): ").strip()
            params['window'] = int(window) if window else 15
            params['polyorder'] = int(poly) if poly else 3
        
        elif method_name == 'exponential':
            alpha = input("Smoothing factor 0-1 (default 0.4): ").strip()
            params['alpha'] = float(alpha) if alpha else 0.4
        
        # Apply filter
        print("\n⏳ Cleaning data...")
        cleaned = clean_data(df, thrust_col, method=method_name, **params)
        
        # Save output
        base_name = os.path.basename(input_file).rsplit('.', 1)[0]
        output_file = f"{base_name}_cleaned.csv"
        
        save_cleaned_data(df, cleaned, thrust_col, output_file, OUTPUT_FOLDER)
        
        # Show statistics
        original = df[thrust_col].values
        print(f"\nStatistics:")
        print(f"  Original - Mean: {np.mean(original):.2f}, Std: {np.std(original):.2f}")
        print(f"  Cleaned  - Mean: {np.mean(cleaned):.2f}, Std: {np.std(cleaned):.2f}")
        print(f"  Noise reduction: {(1 - np.std(cleaned)/np.std(original))*100:.1f}%")
        print(f"  Total Thrust (Original): {np.sum(original):.2f}")
        print(f"  Total Thrust (Cleaned): {np.sum(cleaned):.2f}")
        
        # Generate and display graph
        print("\n⏳ Generating graph...")
        plot_comparison(df, original, cleaned, thrust_col, time_col, method_name, OUTPUT_FOLDER)
    
    else:
        print("✗ Invalid choice")


if __name__ == "__main__":
    main()
