"""
TTEH Batch Analysis Script

Secret key values from paper (used in all evaluations):
  x0 = 0.3271
  mu = 1.9999

Steps:
  1. Check if data/samples/ has images. If empty, generate 80 synthetic
     fingerprints (256x256) with seeds 0-79, save as .png to data/samples/.
  2. Run batch_analyze(data/samples/, x0=0.3271, mu=1.9999).
  3. Print progress per image.
  4. Save results to results/metrics.csv.
  5. Generate 4 matplotlib figures matching paper Figs 6-9 exactly:
       results/plots/fig6_entropy.png     — bar chart, 80 images, horizontal ideal line at 8.0
       results/plots/fig7_npcr.png        — bar chart, 80 images, horizontal ideal line at 99.6093
       results/plots/fig8_uaci.png        — bar chart, 80 images, horizontal ideal line at 33.4635
       results/plots/fig9_correlation.png — horizontal bar chart (matching paper Fig 9 style)
  6. Print final summary table matching paper results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.encryption import generate_synthetic_fingerprint
from src.metrics import batch_analyze


def generate_synthetic_dataset(num_images: int = 80, output_dir: Path = None):
    """
    Generate synthetic fingerprint dataset if data/samples/ is empty.
    
    Args:
        num_images: Number of synthetic images to generate
        output_dir: Directory to save images
    """
    if output_dir is None:
        output_dir = Path("data/samples")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if directory already has images
    existing_images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.bmp"))
    if existing_images:
        print(f"Found {len(existing_images)} existing images in {output_dir}")
        return
    
    print(f"Generating {num_images} synthetic fingerprint images...")
    
    for i in range(num_images):
        img = generate_synthetic_fingerprint(256, 256, seed=i)
        
        # Save as PNG
        filename = f"synthetic_{i:03d}.png"
        filepath = output_dir / filename
        
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img.save(filepath)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print(f"Generated {num_images} synthetic images in {output_dir}")


def create_figure6_entropy(df: pd.DataFrame, output_path: Path):
    """
    Create Figure 6: Entropy bar chart
    - Bar chart, 80 images
    - Horizontal ideal line at 8.0
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    x_pos = np.arange(len(df))
    plt.bar(x_pos, df['entropy'], alpha=0.7, color='blue')
    
    # Add ideal line
    plt.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='Ideal (8.0)')
    
    # Formatting
    plt.xlabel('Image Index')
    plt.ylabel('Entropy')
    plt.title('Information Entropy Analysis (Paper Fig. 6)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show image indices
    plt.xticks(np.arange(0, len(df), 10), np.arange(0, len(df), 10))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_figure7_npcr(df: pd.DataFrame, output_path: Path):
    """
    Create Figure 7: NPCR bar chart
    - Bar chart, 80 images
    - Horizontal ideal line at 99.6093
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    x_pos = np.arange(len(df))
    plt.bar(x_pos, df['npcr'], alpha=0.7, color='green')
    
    # Add ideal line
    plt.axhline(y=99.6093, color='red', linestyle='--', linewidth=2, label='Ideal (99.6093%)')
    
    # Formatting
    plt.xlabel('Image Index')
    plt.ylabel('NPCR (%)')
    plt.title('Number of Pixels Change Rate (Paper Fig. 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(99, 100)
    
    # Set x-axis to show image indices
    plt.xticks(np.arange(0, len(df), 10), np.arange(0, len(df), 10))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_figure8_uaci(df: pd.DataFrame, output_path: Path):
    """
    Create Figure 8: UACI bar chart
    - Bar chart, 80 images
    - Horizontal ideal line at 33.4635
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    x_pos = np.arange(len(df))
    plt.bar(x_pos, df['uaci'], alpha=0.7, color='orange')
    
    # Add ideal line
    plt.axhline(y=33.4635, color='red', linestyle='--', linewidth=2, label='Ideal (33.4635%)')
    
    # Formatting
    plt.xlabel('Image Index')
    plt.ylabel('UACI (%)')
    plt.title('Unified Average Changing Intensity (Paper Fig. 8)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(33, 34)
    
    # Set x-axis to show image indices
    plt.xticks(np.arange(0, len(df), 10), np.arange(0, len(df), 10))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_figure9_correlation(df: pd.DataFrame, output_path: Path):
    """
    Create Figure 9: Correlation horizontal bar chart
    - Horizontal bar chart matching paper Fig 9 style
    - Show mean correlation for three directions
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate mean correlations
    corr_h = df['correlation_h'].mean()
    corr_v = df['correlation_v'].mean()
    corr_d = df['correlation_d'].mean()
    
    # Create horizontal bar chart
    directions = ['Horizontal', 'Vertical', 'Diagonal']
    correlations = [corr_h, corr_v, corr_d]
    
    bars = plt.barh(directions, correlations, alpha=0.7, color=['red', 'green', 'blue'])
    
    # Add value labels on bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{corr:.4f}', ha='left', va='center')
    
    # Add ideal line (near zero)
    plt.axvline(x=0.0036, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Paper Average (0.0036)')
    
    # Formatting
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Direction')
    plt.title('Correlation Analysis (Paper Fig. 9)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.01, max(correlations) + 0.01)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """
    Print final summary table matching paper results.
    """
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE - TTEH SECURITY ANALYSIS")
    print("="*80)
    
    # Calculate statistics
    metrics = {
        'Entropy': {
            'mean': df['entropy'].mean(),
            'std': df['entropy'].std(),
            'paper': 7.9972
        },
        'NPCR': {
            'mean': df['npcr'].mean(),
            'std': df['npcr'].std(),
            'paper': 99.6045
        },
        'UACI': {
            'mean': df['uaci'].mean(),
            'std': df['uaci'].std(),
            'paper': 33.4651
        },
        'Correlation': {
            'mean': df['correlation_mean'].mean(),
            'std': df['correlation_mean'].std(),
            'paper': 0.0036
        }
    }
    
    # Print table
    print(f"{'Metric':<12} | {'Mean':<10} | {'Std':<10} | {'Paper Value':<12} | {'Match?'}")
    print("-" * 65)
    
    for metric, stats in metrics.items():
        if metric in ['NPCR', 'UACI']:
            mean_str = f"{stats['mean']:.2f}%"
            paper_str = f"{stats['paper']:.2f}%"
        else:
            mean_str = f"{stats['mean']:.4f}"
            paper_str = f"{stats['paper']:.4f}"
        
        std_str = f"{stats['std']:.4f}"
        
        # Check if within reasonable range of paper values
        if metric == 'Entropy':
            match = abs(stats['mean'] - stats['paper']) < 0.01
        elif metric == 'NPCR':
            match = abs(stats['mean'] - stats['paper']) < 0.1
        elif metric == 'UACI':
            match = abs(stats['mean'] - stats['paper']) < 0.1
        else:  # Correlation
            match = abs(stats['mean'] - stats['paper']) < 0.01
        
        match_str = "YES" if match else "NO"
        
        print(f"{metric:<12} | {mean_str:<10} | {std_str:<10} | {paper_str:<12} | {match_str}")
    
    print("="*80)


def main():
    """Main analysis script"""
    print("TTEH Fingerprint Image Encryption - Batch Analysis")
    print("=" * 60)
    
    # Secret key values from paper
    x0 = 0.3271
    mu = 1.9999
    
    print(f"Using key parameters: x0 = {x0}, mu = {mu}")
    
    # Step 1: Check/generate dataset
    data_dir = Path("data/samples")
    generate_synthetic_dataset(num_images=80, output_dir=data_dir)
    
    # Step 2: Run batch analysis
    print("\nRunning batch security analysis...")
    df = batch_analyze(data_dir, x0, mu)
    
    # Step 3: Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    csv_path = results_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Step 4: Generate plots
    plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
    print("\nGenerating plots...")
    create_figure6_entropy(df, plots_dir / "fig6_entropy.png")
    create_figure7_npcr(df, plots_dir / "fig7_npcr.png")
    create_figure8_uaci(df, plots_dir / "fig8_uaci.png")
    create_figure9_correlation(df, plots_dir / "fig9_correlation.png")
    print(f"Plots saved to {plots_dir}")
    
    # Step 5: Print summary table
    print_summary_table(df)
    
    print(f"\nAnalysis complete! Processed {len(df)} images.")
    print(f"Results: {csv_path}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
