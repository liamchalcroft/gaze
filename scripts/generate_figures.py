#!/usr/bin/env python3
"""
Generate publication-ready figures from aggregated benchmark results.

This script creates various visualizations including bar charts, radar plots,
and performance comparisons.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import matplotlib.patches as mpatches

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme for approaches
APPROACH_COLORS = {
    "baseline": "#2E86AB",
    "multiturn": "#A23B72", 
    "web_search": "#F18F01",
    "visual": "#C73E1D",
    "comprehensive": "#4C956C"
}

APPROACH_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "comprehensive": "Reasoning + Web + Visual", 
    "multiturn": "Reasoning",
    "visual": "Reasoning + Visual",
    "web_search": "Reasoning + Web"
}

def load_aggregated_results(results_file: Path) -> pd.DataFrame:
    """Load the aggregated results CSV file."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    return pd.read_csv(results_file)

def create_performance_comparison_bar_chart(df: pd.DataFrame, output_file: Path) -> None:
    """Create a bar chart comparing main performance metrics across approaches."""
    
    # Prepare data
    approaches = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]
    metrics_data = {
        'Localization mAP50': [],
        'Caption BLEU': [],
        'Diagnosis Top-1': []
    }
    
    approach_labels = []
    
    for approach in approaches:
        if approach not in df['approach'].values:
            continue
            
        approach_data = df[df['approach'] == approach]
        approach_labels.append(APPROACH_DISPLAY_NAMES.get(approach, approach))
        
        # Get metrics for each task (handle missing tasks)
        loc_data = approach_data[approach_data['task'] == 'localization']
        cap_data = approach_data[approach_data['task'] == 'caption']
        diag_data = approach_data[approach_data['task'] == 'diagnosis']
        
        loc_map50 = loc_data['map50'].mean() if not loc_data.empty and 'map50' in loc_data.columns else float('nan')
        cap_bleu = cap_data['bleu'].mean() if not cap_data.empty and 'bleu' in cap_data.columns else float('nan')
        diag_top1 = diag_data['top1'].mean() if not diag_data.empty and 'top1' in diag_data.columns else float('nan')
        
        metrics_data['Localization mAP50'].append(loc_map50 * 100 if not pd.isna(loc_map50) else 0)
        metrics_data['Caption BLEU'].append(cap_bleu if not pd.isna(cap_bleu) else 0)
        metrics_data['Diagnosis Top-1'].append(diag_top1 * 100 if not pd.isna(diag_top1) else 0)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Performance Comparison Across Tasks', fontsize=16, fontweight='bold')
    
    # Plot each metric
    metric_names = list(metrics_data.keys())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        bars = ax.bar(approach_labels, values, color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add directional arrow to indicate higher is better
        title_with_arrow = f"{metric} ↑"
        ax.set_title(title_with_arrow, fontweight='bold')
        ax.set_ylabel('Score' if 'BLEU' in metric else 'Percentage (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.1f}' if 'BLEU' in metric else f'{value:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, max(values) * 1.2 if values else 100)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance comparison chart saved to {output_file}")

def create_radar_plot(df: pd.DataFrame, output_file: Path) -> None:
    """Create a radar plot showing multi-dimensional performance."""
    
    # Prepare data for radar plot
    approaches = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]
    metrics = ['Localization mAP50', 'Caption BLEU', 'Caption METEOR', 'Diagnosis Top1', 'Diagnosis Top5']
    
    # Collect data
    radar_data = {}
    for approach in approaches:
        if approach not in df['approach'].values:
            continue
            
        approach_data = df[df['approach'] == approach]
        
        loc_data = approach_data[approach_data['task'] == 'localization']
        cap_data = approach_data[approach_data['task'] == 'caption']
        diag_data = approach_data[approach_data['task'] == 'diagnosis']
        
        values = [
            loc_data['map50'].mean() * 100 if not loc_data.empty else 0,
            cap_data['bleu'].mean() * 100 if not cap_data.empty else 0,
            cap_data['meteor'].mean() * 100 if not cap_data.empty else 0,
            diag_data['top1'].mean() * 100 if not diag_data.empty else 0,
            diag_data['top5'].mean() * 100 if not diag_data.empty else 0,
        ]
        
        radar_data[approach] = values
    
    # Set up the radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each approach
    for approach, values in radar_data.items():
        values += values[:1]  # Complete the circle
        color = APPROACH_COLORS.get(approach, '#000000')
        label = APPROACH_DISPLAY_NAMES.get(approach, approach)
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Customize the plot with directional arrows (all these metrics: higher is better)
    metrics_with_arrows = [f"{metric} ↑" for metric in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_with_arrows, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    plt.title('Multi-Dimensional Performance Comparison', size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Radar plot saved to {output_file}")

def create_task_specific_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create task-specific comparison plots."""
    
    tasks = ['localization', 'caption', 'diagnosis']
    
    for task in tasks:
        task_data = df[df['task'] == task]
        if task_data.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data based on task
        if task == 'localization':
            metrics = ['map30', 'map50', 'map50_95']
            metric_labels = ['mAP@30 ↑', 'mAP@50 ↑', 'mAP@50:95 ↑']
            ylabel = 'Mean Average Precision (%)'
            title = 'Localization Performance (Object Detection Metrics)'
        elif task == 'caption':
            metrics = ['bleu', 'meteor', 'bert_f1', 'radgraph_f1']
            metric_labels = ['BLEU ↑', 'METEOR ↑', 'BERT-F1 ↑', 'RadGraph-F1 ↑']
            ylabel = 'Score'
            title = 'Caption Generation Performance'
        else:  # diagnosis
            metrics = ['top1', 'top5', 'coverage', 'entropy']
            metric_labels = ['Top-1 ↑', 'Top-5 ↑', 'Coverage ↑', 'Entropy ↓']
            ylabel = 'Score'
            title = 'Diagnosis Classification Performance (GPT-4o Semantic Matching)'
        
        # Create grouped bar chart - use consistent ordering
        all_approaches = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]
        available_approaches = [app for app in all_approaches if app in task_data['approach'].values]
        
        x = np.arange(len(metric_labels))
        width = 0.15
        
        for i, approach in enumerate(available_approaches):
            if approach not in APPROACH_COLORS:
                continue
                
            approach_task_data = task_data[task_data['approach'] == approach]
            values = []
            
            for metric in metrics:
                if metric in approach_task_data.columns:
                    val = approach_task_data[metric].mean()
                    if task == 'caption' and metric == 'bleu':
                        values.append(val if not pd.isna(val) else 0)
                    else:
                        values.append(val * 100 if not pd.isna(val) else 0)
                else:
                    values.append(0)
            
            color = APPROACH_COLORS[approach]
            label = APPROACH_DISPLAY_NAMES.get(approach, approach)
            
            bars = ax.bar(x + i * width, values, width, label=label, color=color, 
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * (len(available_approaches) - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.legend()  # Legend will be in correct order since we iterate approaches correctly
        ax.grid(True, alpha=0.3)
        
        output_file = output_dir / f"{task}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"{task.title()} comparison saved to {output_file}")

def create_sample_count_visualization(df: pd.DataFrame, output_file: Path) -> None:
    """Create a visualization showing sample counts and success rates."""
    
    # Aggregate by approach
    approach_stats = df.groupby('approach').agg({
        'sample_count': 'sum',
        'failed_samples': 'sum'
    }).reset_index()
    
    approach_stats['success_rate'] = (
        (approach_stats['sample_count'] - approach_stats['failed_samples']) / 
        approach_stats['sample_count'] * 100
    )
    
    # Filter to known approaches
    approach_stats = approach_stats[approach_stats['approach'].isin(APPROACH_COLORS.keys())]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sample counts - use consistent ordering
    all_approaches = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]
    available_approaches = [app for app in all_approaches if app in approach_stats['approach'].values]
    
    # Reorder approach_stats to match our standard order
    approach_stats = approach_stats.set_index('approach').reindex(available_approaches).reset_index()
    
    approaches = approach_stats['approach'].tolist()
    colors = [APPROACH_COLORS[app] for app in approaches]
    labels = [APPROACH_DISPLAY_NAMES.get(app, app) for app in approaches]
    
    bars1 = ax1.bar(labels, approach_stats['sample_count'], color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax1.set_title('Total Samples Processed', fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, approach_stats['sample_count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # Success rates
    bars2 = ax2.bar(labels, approach_stats['success_rate'], color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.5)
    ax2.set_title('Success Rate', fontweight='bold')
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 105)
    
    # Add value labels
    for bar, value in zip(bars2, approach_stats['success_rate']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sample count visualization saved to {output_file}")

def create_heatmap(df: pd.DataFrame, output_file: Path) -> None:
    """Create a heatmap showing performance across approaches and tasks."""
    
    # Prepare data for heatmap
    approaches = ["baseline", "multiturn", "web_search", "visual", "comprehensive"]
    tasks = ['localization', 'caption', 'diagnosis']
    
    # Create matrix
    data_matrix = []
    approach_labels = []
    
    for approach in approaches:
        if approach not in df['approach'].values:
            continue
            
        approach_data = df[df['approach'] == approach]
        row = []
        
        for task in tasks:
            task_data = approach_data[approach_data['task'] == task]
            
            if task == 'localization':
                value = task_data['map50'].mean() * 100 if not task_data.empty and 'map50' in task_data.columns else 0
            elif task == 'caption':
                value = task_data['bleu'].mean() * 100 if not task_data.empty and 'bleu' in task_data.columns else 0
            else:  # diagnosis
                value = task_data['top1'].mean() * 100 if not task_data.empty and 'top1' in task_data.columns else 0
            
            row.append(value if not pd.isna(value) else 0)
        
        data_matrix.append(row)
        approach_labels.append(APPROACH_DISPLAY_NAMES.get(approach, approach))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(approach_labels)))
    ax.set_xticklabels([t.title() for t in tasks])
    ax.set_yticklabels(approach_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Performance Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(approach_labels)):
        for j in range(len(tasks)):
            text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Heatmap Across Tasks and Approaches', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance heatmap saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate figures from benchmark results")
    parser.add_argument(
        "--results-file",
        type=str,
        default="outputs/results_analysis/aggregated_results.csv",
        help="Path to aggregated results CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figures"
    )
    parser.add_argument(
        "--figure-type",
        choices=["all", "bar", "radar", "task", "samples", "heatmap"],
        default="all",
        help="Type of figure to generate"
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for figures"
    )
    
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from {results_file}")
    df = load_aggregated_results(results_file)
    
    logger.info(f"Generating figures with format: {args.format}")
    
    # Generate figures based on requested type
    if args.figure_type in ["all", "bar"]:
        create_performance_comparison_bar_chart(
            df, output_dir / f"performance_comparison.{args.format}"
        )
    
    if args.figure_type in ["all", "radar"]:
        create_radar_plot(df, output_dir / f"radar_plot.{args.format}")
    
    if args.figure_type in ["all", "task"]:
        create_task_specific_comparison(df, output_dir)
    
    if args.figure_type in ["all", "samples"]:
        create_sample_count_visualization(
            df, output_dir / f"sample_counts.{args.format}"
        )
    
    if args.figure_type in ["all", "heatmap"]:
        create_heatmap(df, output_dir / f"performance_heatmap.{args.format}")
    
    logger.info(f"Figures generated in {output_dir}")

if __name__ == "__main__":
    main() 