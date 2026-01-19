"""
Example usage of raga information from the processed metadata.

This script demonstrates various ways to work with raga data.
"""

import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def load_metadata(csv_path='processed/metadata.csv'):
    """Load the processed metadata CSV."""
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found. Run the processing pipeline first.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} audio clips from metadata")
    return df


def example_1_basic_filtering():
    """Example 1: Basic filtering by raga."""
    print("\n" + "="*80)
    print("Example 1: Basic Filtering by Raga")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Filter for a specific raga
    raga_name = 'Yaman'
    yaman_clips = df[df['raga'] == raga_name]
    
    print(f"\nFound {len(yaman_clips)} clips in {raga_name} raga")
    
    if len(yaman_clips) > 0:
        print(f"\nSample {raga_name} clips:")
        for idx, row in yaman_clips.head(5).iterrows():
            print(f"  - {row['datapoint']}")
            print(f"    Duration: {row['duration']:.2f}s, Dataset: {row['dataset_source']}")


def example_2_raga_statistics():
    """Example 2: Analyze raga distribution."""
    print("\n" + "="*80)
    print("Example 2: Raga Statistics")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Count clips per raga
    raga_counts = df['raga'].value_counts()
    
    print(f"\nTotal unique ragas: {len(raga_counts)}")
    print(f"\nTop 10 ragas by clip count:")
    for raga, count in raga_counts.head(10).items():
        print(f"  {raga:20s}: {count:4d} clips")
    
    # Calculate total duration per raga
    raga_duration = df.groupby('raga')['duration'].sum().sort_values(ascending=False)
    
    print(f"\nTop 10 ragas by total duration:")
    for raga, duration in raga_duration.head(10).items():
        hours = duration / 3600
        print(f"  {raga:20s}: {hours:6.2f} hours")


def example_3_dataset_analysis():
    """Example 3: Analyze ragas by dataset."""
    print("\n" + "="*80)
    print("Example 3: Ragas by Dataset")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Group by dataset
    for dataset in df['dataset_source'].unique():
        dataset_df = df[df['dataset_source'] == dataset]
        unique_ragas = dataset_df['raga'].nunique()
        total_clips = len(dataset_df)
        
        print(f"\n{dataset}:")
        print(f"  Total clips: {total_clips}")
        print(f"  Unique ragas: {unique_ragas}")
        
        # Show top ragas in this dataset
        top_ragas = dataset_df['raga'].value_counts().head(5)
        print(f"  Top ragas:")
        for raga, count in top_ragas.items():
            print(f"    - {raga}: {count} clips")


def example_4_thaat_analysis():
    """Example 4: Analyze thaats (Hindustani parent scales)."""
    print("\n" + "="*80)
    print("Example 4: Thaat Analysis (Hindustani Music)")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Filter for clips with thaat information
    hindustani = df[df['thaat'] != '']
    
    if len(hindustani) == 0:
        print("\nNo thaat information found in metadata.")
        print("Thaat is only available for ThaatRagaForest dataset (06).")
        return
    
    print(f"\nFound {len(hindustani)} clips with thaat information")
    
    # Group by thaat
    for thaat in hindustani['thaat'].unique():
        thaat_df = hindustani[hindustani['thaat'] == thaat]
        ragas = thaat_df['raga'].unique()
        
        print(f"\n{thaat} Thaat:")
        print(f"  Ragas: {', '.join(ragas)}")
        print(f"  Total clips: {len(thaat_df)}")


def example_5_create_training_dataset():
    """Example 5: Create a training dataset for specific ragas."""
    print("\n" + "="*80)
    print("Example 5: Create Training Dataset")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Select specific ragas for training
    target_ragas = ['Yaman', 'Bhairavi', 'Bhupali', 'Kalyani']
    
    training_data = df[df['raga'].isin(target_ragas)]
    
    print(f"\nCreating training dataset with ragas: {', '.join(target_ragas)}")
    print(f"Total clips: {len(training_data)}")
    
    # Show distribution
    print(f"\nDistribution:")
    for raga in target_ragas:
        count = len(training_data[training_data['raga'] == raga])
        print(f"  {raga:15s}: {count:4d} clips")
    
    # Save to new CSV
    output_path = 'processed/training_dataset.csv'
    training_data.to_csv(output_path, index=False)
    print(f"\nSaved training dataset to: {output_path}")


def example_6_vocal_vs_instrumental():
    """Example 6: Analyze vocal vs instrumental by raga."""
    print("\n" + "="*80)
    print("Example 6: Vocal vs Instrumental by Raga")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Get top ragas
    top_ragas = df['raga'].value_counts().head(5).index
    
    print(f"\nVocal vs Instrumental distribution for top ragas:")
    
    for raga in top_ragas:
        raga_df = df[df['raga'] == raga]
        vocal_count = len(raga_df[raga_df['vocal_instrumental'] == 'vocal'])
        instrumental_count = len(raga_df[raga_df['vocal_instrumental'] == 'instrumental'])
        
        print(f"\n{raga}:")
        print(f"  Vocal: {vocal_count} clips")
        print(f"  Instrumental: {instrumental_count} clips")


def example_7_plot_raga_distribution():
    """Example 7: Visualize raga distribution."""
    print("\n" + "="*80)
    print("Example 7: Visualize Raga Distribution")
    print("="*80)
    
    df = load_metadata()
    if df is None:
        return
    
    # Get top 15 ragas
    raga_counts = df['raga'].value_counts().head(15)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    raga_counts.plot(kind='bar', color='steelblue')
    plt.title('Top 15 Ragas by Clip Count', fontsize=14, fontweight='bold')
    plt.xlabel('Raga', fontsize=12)
    plt.ylabel('Number of Clips', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = 'processed/raga_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Show plot
    try:
        plt.show()
    except:
        print("(Display not available, plot saved to file)")


def main():
    """Run all examples."""
    print("="*80)
    print("Raga Usage Examples")
    print("="*80)
    print("\nThis script demonstrates various ways to work with raga information")
    print("from the processed metadata CSV.\n")
    
    examples = [
        ("1", "Basic filtering by raga", example_1_basic_filtering),
        ("2", "Raga statistics", example_2_raga_statistics),
        ("3", "Ragas by dataset", example_3_dataset_analysis),
        ("4", "Thaat analysis", example_4_thaat_analysis),
        ("5", "Create training dataset", example_5_create_training_dataset),
        ("6", "Vocal vs instrumental", example_6_vocal_vs_instrumental),
        ("7", "Plot distribution", example_7_plot_raga_distribution),
    ]
    
    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    print("  all. Run all examples")
    print("  q. Quit")
    
    while True:
        choice = input("\nSelect example (1-7, all, or q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            for _, _, func in examples:
                try:
                    func()
                except Exception as e:
                    print(f"Error: {e}")
            break
        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            idx = int(choice) - 1
            try:
                examples[idx][2]()
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice. Please enter 1-7, 'all', or 'q'.")


if __name__ == "__main__":
    main()
