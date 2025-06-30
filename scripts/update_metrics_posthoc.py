#!/usr/bin/env python3
"""
Post-hoc metrics update script for NOVA Retrieval VLM.

This script allows re-evaluating existing benchmark results to update metrics JSONs.
This is particularly useful for applying updated evaluation methods (like the new
GPT-4o semantic matching for diagnosis) without re-running the entire benchmark.

Usage:
    python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark
    python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark --task diagnosis --approach baseline
    python scripts/update_metrics_posthoc.py --results-dir outputs_test --force
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import traceback
from dataclasses import dataclass
from loguru import logger

# Import evaluation functions
from nova_retrieval_vlm.evaluation import evaluate


@dataclass
class ResultDirectory:
    """Information about a result directory."""
    path: Path
    approach: str
    task: str
    model: str
    has_consolidated_files: bool
    individual_dirs: List[Path]
    metrics_file: Optional[Path] = None


def discover_result_directories(base_dir: Path, 
                              approach_filter: Optional[str] = None,
                              task_filter: Optional[str] = None,
                              model_filter: Optional[str] = None) -> List[ResultDirectory]:
    """Discover all result directories that can be re-evaluated."""
    result_dirs = []
    
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        return result_dirs
    
    logger.info(f"Scanning for result directories in: {base_dir}")
    
    # Look for the standard benchmark structure: approach/task/model/timestamp/
    for approach_dir in base_dir.iterdir():
        if not approach_dir.is_dir():
            continue
        
        approach = approach_dir.name
        if approach_filter and approach != approach_filter:
            continue
            
        for task_dir in approach_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            task = task_dir.name
            if task_filter and task != task_filter:
                continue
                
            for model_dir in task_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model = model_dir.name
                if model_filter and model != model_filter:
                    continue
                
                # Look for timestamp directories or direct result files
                timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                
                for timestamp_dir in timestamp_dirs:
                    # Check if this directory has consolidated preds.jsonl and refs.jsonl
                    preds_file = timestamp_dir / "preds.jsonl"
                    refs_file = timestamp_dir / "refs.jsonl"
                    
                    # Get individual image directories
                    individual_dirs = [d for d in timestamp_dir.iterdir() 
                                     if d.is_dir() and d.name.startswith("image_")]
                    
                    if preds_file.exists() and refs_file.exists():
                        # Has consolidated files
                        result_dir = ResultDirectory(
                            path=timestamp_dir,
                            approach=approach,
                            task=task,
                            model=model,
                            has_consolidated_files=True,
                            individual_dirs=individual_dirs
                        )
                        result_dirs.append(result_dir)
                        logger.debug(f"Found consolidated result: {result_dir.path}")
                        
                    elif individual_dirs:
                        # Has individual directories only
                        result_dir = ResultDirectory(
                            path=timestamp_dir,
                            approach=approach,
                            task=task,
                            model=model,
                            has_consolidated_files=False,
                            individual_dirs=individual_dirs
                        )
                        result_dirs.append(result_dir)
                        logger.debug(f"Found individual result: {result_dir.path}")
    
    # Also check for direct result directories (like outputs_test structure)
    for item in base_dir.iterdir():
        if item.is_dir() and not any(item.name == rd.approach for rd in result_dirs):
            # Check if this looks like a direct results directory
            potential_files = list(item.glob("**/*.jsonl"))
            if potential_files:
                # This might be a different structure, add as generic
                result_dir = ResultDirectory(
                    path=item,
                    approach="unknown",
                    task="unknown",
                    model="unknown",
                    has_consolidated_files=False,
                    individual_dirs=[]
                )
                result_dirs.append(result_dir)
                logger.debug(f"Found generic result: {result_dir.path}")
    
    logger.info(f"Found {len(result_dirs)} result directories")
    return result_dirs


def backup_metrics_file(metrics_file: Path) -> Path:
    """Create a backup of existing metrics file."""
    if not metrics_file.exists():
        return metrics_file
        
    backup_file = metrics_file.parent / f"{metrics_file.stem}_backup_{int(time.time())}{metrics_file.suffix}"
    backup_file.write_text(metrics_file.read_text())
    logger.debug(f"Backed up metrics to: {backup_file}")
    return backup_file


def update_consolidated_metrics(result_dir: ResultDirectory, force: bool = False) -> bool:
    """Update metrics for a directory with consolidated preds.jsonl and refs.jsonl files."""
    preds_file = result_dir.path / "preds.jsonl"
    refs_file = result_dir.path / "refs.jsonl"
    
    if not (preds_file.exists() and refs_file.exists()):
        logger.warning(f"Missing consolidated files in {result_dir.path}")
        return False
    
    # Look for existing aggregated metrics
    possible_metrics_files = [
        result_dir.path / "metrics.json",
        result_dir.path / "run.json",
        result_dir.path / "results.json"
    ]
    
    existing_metrics_file = None
    for mf in possible_metrics_files:
        if mf.exists():
            existing_metrics_file = mf
            break
    
    if existing_metrics_file and not force:
        logger.info(f"Metrics already exist at {existing_metrics_file}, skipping (use --force to override)")
        return False
    
    try:
        logger.info(f"Re-evaluating consolidated metrics for {result_dir.approach}/{result_dir.task}/{result_dir.model}")
        
        # Backup existing metrics if they exist
        if existing_metrics_file:
            backup_metrics_file(existing_metrics_file)
        
        # Run evaluation
        start_time = time.time()
        new_metrics = evaluate(str(preds_file), str(refs_file), task=result_dir.task)
        evaluation_time = time.time() - start_time
        
        # Save updated metrics
        metrics_file = result_dir.path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(new_metrics, f, indent=2)
        
        logger.success(f"Updated consolidated metrics in {evaluation_time:.2f}s: {metrics_file}")
        logger.info(f"New metrics: {new_metrics}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update consolidated metrics for {result_dir.path}: {e}")
        logger.debug(traceback.format_exc())
        return False


def update_individual_metrics(result_dir: ResultDirectory, force: bool = False) -> Tuple[int, int]:
    """Update metrics for individual image directories. Returns (success_count, total_count)."""
    if not result_dir.individual_dirs:
        logger.warning(f"No individual directories found in {result_dir.path}")
        return 0, 0
    
    success_count = 0
    total_count = len(result_dir.individual_dirs)
    
    logger.info(f"Re-evaluating {total_count} individual samples for {result_dir.approach}/{result_dir.task}/{result_dir.model}")
    
    for i, img_dir in enumerate(result_dir.individual_dirs):
        pred_file = img_dir / "pred.jsonl"
        ref_file = img_dir / "ref.jsonl"
        metrics_file = img_dir / "metrics.json"
        
        if not (pred_file.exists() and ref_file.exists()):
            logger.warning(f"Missing pred/ref files in {img_dir}")
            continue
        
        if metrics_file.exists() and not force:
            logger.debug(f"Metrics already exist for {img_dir.name}, skipping")
            success_count += 1
            continue
        
        try:
            # Backup existing metrics if they exist
            if metrics_file.exists():
                backup_metrics_file(metrics_file)
            
            # Run evaluation for this individual sample
            new_metrics = evaluate(str(pred_file), str(ref_file), task=result_dir.task)
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(new_metrics, f, indent=2)
            
            success_count += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total_count} samples processed")
            
        except Exception as e:
            logger.error(f"Failed to update metrics for {img_dir}: {e}")
            logger.debug(traceback.format_exc())
    
    logger.info(f"Updated {success_count}/{total_count} individual metrics")
    return success_count, total_count


def aggregate_individual_metrics(result_dir: ResultDirectory) -> Optional[Dict]:
    """Aggregate metrics from individual directories into a consolidated metrics.json."""
    if not result_dir.individual_dirs:
        return None
    
    all_metrics = []
    
    for img_dir in result_dir.individual_dirs:
        metrics_file = img_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                all_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
    
    if not all_metrics:
        logger.warning(f"No valid individual metrics found in {result_dir.path}")
        return None
    
    # Aggregate metrics by averaging
    aggregated = {}
    for key in all_metrics[0].keys():
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if values:
            if isinstance(values[0], (int, float)):
                aggregated[key] = sum(values) / len(values)
            else:
                # For non-numeric values, take the most common
                aggregated[key] = max(set(values), key=values.count)
    
    # Add metadata
    aggregated['_meta'] = {
        'sample_count': len(all_metrics),
        'aggregation_method': 'mean',
        'timestamp': time.time()
    }
    
    # Save aggregated metrics
    agg_metrics_file = result_dir.path / "aggregated_metrics.json"
    with open(agg_metrics_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    logger.info(f"Aggregated metrics saved to: {agg_metrics_file}")
    return aggregated


def main():
    """Main function for post-hoc metrics update."""
    parser = argparse.ArgumentParser(
        description="Update evaluation metrics for existing benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all results
  python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark
  
  # Update only diagnosis results
  python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark --task diagnosis
  
  # Update specific approach and force overwrite
  python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark --approach baseline --force
  
  # Update with specific filters
  python scripts/update_metrics_posthoc.py --results-dir runs/full_benchmark --task diagnosis --model "google_gemini-2.5-flash-preview-05-20"
        """)
    
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Base directory containing benchmark results")
    parser.add_argument("--approach", type=str, 
                       help="Filter by approach (baseline, multiturn, web_search, visual, comprehensive)")
    parser.add_argument("--task", type=str,
                       help="Filter by task (localization, caption, diagnosis)")
    parser.add_argument("--model", type=str,
                       help="Filter by model name")
    parser.add_argument("--force", action="store_true",
                       help="Force update even if metrics already exist")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing individual metrics, don't re-evaluate")
    parser.add_argument("--consolidate-only", action="store_true",
                       help="Only update consolidated metrics, skip individual samples")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    # Discover result directories
    logger.info(f"Discovering result directories in: {results_dir}")
    result_directories = discover_result_directories(
        results_dir, 
        approach_filter=args.approach,
        task_filter=args.task,
        model_filter=args.model
    )
    
    if not result_directories:
        logger.error("No result directories found matching the criteria")
        sys.exit(1)
    
    logger.info(f"Found {len(result_directories)} result directories to process")
    
    # Process each directory
    total_success = 0
    total_processed = 0
    
    for result_dir in result_directories:
        logger.info(f"\nProcessing: {result_dir.path}")
        logger.info(f"  Approach: {result_dir.approach}")
        logger.info(f"  Task: {result_dir.task}")
        logger.info(f"  Model: {result_dir.model}")
        logger.info(f"  Has consolidated files: {result_dir.has_consolidated_files}")
        logger.info(f"  Individual directories: {len(result_dir.individual_dirs)}")
        
        try:
            if args.aggregate_only:
                # Only aggregate existing metrics
                if result_dir.individual_dirs:
                    aggregated = aggregate_individual_metrics(result_dir)
                    if aggregated:
                        total_success += 1
                else:
                    logger.warning("No individual directories to aggregate")
                    
            elif result_dir.has_consolidated_files and not args.consolidate_only:
                # Update consolidated metrics
                if update_consolidated_metrics(result_dir, force=args.force):
                    total_success += 1
                    
            elif result_dir.individual_dirs:
                # Update individual metrics
                success_count, total_count = update_individual_metrics(result_dir, force=args.force)
                if success_count > 0:
                    total_success += 1
                    
                    # Also create aggregated metrics
                    aggregate_individual_metrics(result_dir)
            else:
                logger.warning("No evaluable files found in directory")
                
        except Exception as e:
            logger.error(f"Failed to process {result_dir.path}: {e}")
            logger.debug(traceback.format_exc())
        
        total_processed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Post-hoc metrics update completed!")
    logger.info(f"Successfully processed: {total_success}/{total_processed} directories")
    
    if total_success > 0:
        logger.info(f"\nTo regenerate analysis tables and figures, run:")
        logger.info(f"  python scripts/gather_results.py --results-dir {results_dir}")
        logger.info(f"  python scripts/generate_latex_tables.py")
        logger.info(f"  python scripts/generate_figures.py")
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 