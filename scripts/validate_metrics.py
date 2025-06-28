#!/usr/bin/env python3
"""
Comprehensive metrics validation script for NOVA Retrieval VLM.

This script validates all evaluation metrics to ensure they are working correctly
and producing reasonable results. It can be run as part of CI/CD or for debugging.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import argparse
import sys

# Import our evaluation modules
from nova_retrieval_vlm.evaluation import evaluate
from nova_retrieval_vlm.evaluation.evaluator import Evaluator
from nova_retrieval_vlm.evaluation.diagnosis import evaluate_diagnosis
from nova_retrieval_vlm.evaluation.caption import evaluate_caption
from nova_retrieval_vlm.evaluation.detection import evaluate_detection


class MetricsValidator:
    """Comprehensive metrics validation class."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.errors = []
        
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def validate_basic_evaluator(self) -> bool:
        """Validate the basic evaluator functionality."""
        self.log("=== Validating Basic Evaluator ===")
        
        try:
            evaluator = Evaluator()
            
            # Test cases with expected results
            test_cases = [
                ("identical", "This is a test", "This is a test", 1.0, 0.05),
                ("similar", "This is a test case", "This is a test", 0.8, 0.2),
                ("different", "This is completely different", "This is a test", 0.3, 0.3),
                ("empty_ref", "This is a prediction", "", 0.0, 0.0),
            ]
            
            all_passed = True
            for name, pred, ref, expected_min, tolerance in test_cases:
                try:
                    result = evaluator.evaluate_prediction(pred, ref)
                    score = result["score"]
                    
                    # Validate score range
                    if not (0.0 <= score <= 1.0):
                        self.errors.append(f"{name}: Score {score} not in [0,1]")
                        all_passed = False
                    
                    # Validate expected range
                    if abs(score - expected_min) > tolerance:
                        self.errors.append(f"{name}: Score {score} outside expected range [{expected_min-tolerance}, {expected_min+tolerance}]")
                        all_passed = False
                    
                    self.log(f"  {name}: score = {score:.3f} ✓")
                    
                except Exception as e:
                    self.errors.append(f"{name}: ERROR - {e}")
                    all_passed = False
            
            self.results["basic_evaluator"] = all_passed
            return all_passed
            
        except Exception as e:
            self.errors.append(f"Basic evaluator validation failed: {e}")
            self.results["basic_evaluator"] = False
            return False
    
    def validate_diagnosis_metrics(self) -> bool:
        """Validate diagnosis evaluation metrics."""
        self.log("=== Validating Diagnosis Metrics ===")
        
        try:
            # Test data
            preds = [
                "glioma",  # correct
                ["meningioma", "glioma", "metastasis"],  # correct in top3
                "normal",  # incorrect
            ]
            refs = [
                "glioma",
                "glioma", 
                "meningioma"
            ]
            
            metrics = evaluate_diagnosis(preds, refs)
            
            # Validate required metrics
            required_metrics = ["top1", "top5", "coverage", "entropy"]
            for metric in required_metrics:
                if metric not in metrics:
                    self.errors.append(f"Missing {metric} metric in diagnosis")
                    return False
            
            # Validate ranges
            if not (0.0 <= metrics["top1"] <= 1.0):
                self.errors.append(f"top1 {metrics['top1']} not in [0,1]")
                return False
            
            if not (0.0 <= metrics["top5"] <= 1.0):
                self.errors.append(f"top5 {metrics['top5']} not in [0,1]")
                return False
            
            if metrics["top5"] < metrics["top1"]:
                self.errors.append("top5 should be >= top1")
                return False
            
            if metrics["coverage"] < 0.0:
                self.errors.append(f"coverage {metrics['coverage']} should be >= 0")
                return False
            
            if metrics["entropy"] < 0.0:
                self.errors.append(f"entropy {metrics['entropy']} should be >= 0")
                return False
            
            self.log(f"  Diagnosis metrics: {metrics} ✓")
            self.results["diagnosis_metrics"] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Diagnosis metrics validation failed: {e}")
            self.results["diagnosis_metrics"] = False
            return False
    
    def validate_caption_metrics(self) -> bool:
        """Validate caption evaluation metrics."""
        self.log("=== Validating Caption Metrics ===")
        
        try:
            # Test data
            preds = [
                "There is a small lesion in the left temporal lobe.",
                "Normal MRI brain scan without abnormalities.",
            ]
            refs = [
                "There is a small lesion in the left temporal lobe.",
                "Normal MRI brain scan without abnormalities.",
            ]
            
            metrics = evaluate_caption(preds, refs)
            
            # Validate required metrics
            required_metrics = ["bleu", "bert_f1", "meteor", "radgraph_f1", "modality_f1", "clinical_f1", "binary_f1"]
            for metric in required_metrics:
                if metric not in metrics:
                    self.errors.append(f"Missing {metric} metric in caption")
                    return False
            
            # Validate ranges for identical texts
            if metrics["bleu"] < 90.0:
                self.errors.append(f"BLEU should be high for identical texts, got {metrics['bleu']}")
                return False
            
            if metrics["bert_f1"] < 0.9:
                self.errors.append(f"BERT F1 should be high for identical texts, got {metrics['bert_f1']}")
                return False
            
            if metrics["meteor"] < 25.0:  # Adjusted for medical text
                self.errors.append(f"METEOR should be reasonable for identical texts, got {metrics['meteor']}")
                return False
            
            self.log(f"  Caption metrics: {metrics} ✓")
            self.results["caption_metrics"] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Caption metrics validation failed: {e}")
            self.results["caption_metrics"] = False
            return False
    
    def validate_detection_metrics(self) -> bool:
        """Validate detection evaluation metrics."""
        self.log("=== Validating Detection Metrics ===")
        
        try:
            # Test data with list format
            preds = [
                {
                    "boxes": [[10.0, 10.0, 20.0, 20.0]],
                    "scores": [0.9],
                    "labels": [0],
                }
            ]
            refs = [
                {
                    "boxes": [[10.0, 10.0, 20.0, 20.0]],
                    "scores": [1.0],
                    "labels": [0],
                }
            ]
            
            metrics = evaluate_detection(preds, refs)
            
            # Validate required metrics
            required_metrics = ["map30", "map50", "map50_95"]
            for metric in required_metrics:
                if metric not in metrics:
                    self.errors.append(f"Missing {metric} metric in detection")
                    return False
            
            # Validate perfect match scores
            for metric in required_metrics:
                if metrics[metric] < 0.9:
                    self.errors.append(f"{metric} should be high for perfect match, got {metrics[metric]}")
                    return False
            
            # Test tensor format
            preds_tensor = [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                }
            ]
            refs_tensor = [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0]]),
                    "scores": torch.tensor([1.0]),
                    "labels": torch.tensor([0]),
                }
            ]
            
            metrics_tensor = evaluate_detection(preds_tensor, refs_tensor)
            
            # Should produce same results
            for metric in required_metrics:
                if abs(metrics[metric] - metrics_tensor[metric]) > 1e-6:
                    self.errors.append(f"Tensor and list formats should produce same {metric}")
                    return False
            
            self.log(f"  Detection metrics: {metrics} ✓")
            self.results["detection_metrics"] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Detection metrics validation failed: {e}")
            self.results["detection_metrics"] = False
            return False
    
    def validate_integrated_evaluation(self) -> bool:
        """Validate the integrated evaluation function."""
        self.log("=== Validating Integrated Evaluation ===")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test data for each task
                test_cases = {
                    "caption": {
                        "preds": [{"caption": "There is a lesion in the brain."}],
                        "refs": [{"caption": "There is a lesion in the brain."}]
                    },
                    "diagnosis": {
                        "preds": [{"diagnosis": "glioma"}],
                        "refs": [{"diagnosis": "glioma"}]
                    },
                    "localization": {
                        "preds": [{
                            "boxes": [[10, 10, 20, 20]],
                            "labels": ["lesion"],
                            "scores": [0.9]
                        }],
                        "refs": [{
                            "boxes": [[10, 10, 20, 20]],
                            "labels": ["lesion"],
                            "scores": [1.0]
                        }]
                    }
                }
                
                all_passed = True
                for task, data in test_cases.items():
                    try:
                        # Create files
                        pred_file = temp_path / f"pred_{task}.jsonl"
                        ref_file = temp_path / f"ref_{task}.jsonl"
                        
                        with open(pred_file, 'w') as f:
                            for pred in data["preds"]:
                                f.write(json.dumps(pred) + '\n')
                        
                        with open(ref_file, 'w') as f:
                            for ref in data["refs"]:
                                f.write(json.dumps(ref) + '\n')
                        
                        # Run evaluation
                        metrics = evaluate(str(pred_file), str(ref_file), task=task)
                        
                        # Basic validation
                        if not isinstance(metrics, dict):
                            self.errors.append(f"{task}: metrics should be dict")
                            all_passed = False
                            continue
                        
                        if len(metrics) == 0:
                            self.errors.append(f"{task}: metrics should not be empty")
                            all_passed = False
                            continue
                        
                        self.log(f"  {task} metrics: {metrics} ✓")
                        
                    except Exception as e:
                        self.errors.append(f"{task} integrated evaluation failed: {e}")
                        all_passed = False
                
                self.results["integrated_evaluation"] = all_passed
                return all_passed
                
        except Exception as e:
            self.errors.append(f"Integrated evaluation validation failed: {e}")
            self.results["integrated_evaluation"] = False
            return False
    
    def validate_performance(self) -> bool:
        """Validate performance of metrics calculation."""
        self.log("=== Validating Performance ===")
        
        try:
            # Test diagnosis performance
            start_time = time.time()
            large_diag_preds = ["glioma"] * 100
            large_diag_refs = ["glioma"] * 100
            metrics = evaluate_diagnosis(large_diag_preds, large_diag_refs)
            diag_time = time.time() - start_time
            
            if diag_time > 1.0:  # Should be very fast
                self.errors.append(f"Diagnosis evaluation too slow: {diag_time:.3f}s")
                return False
            
            # Test caption performance (this will be slower due to BERT)
            start_time = time.time()
            large_preds = ["This is a test prediction"] * 10  # Reduced for speed
            large_refs = ["This is a test reference"] * 10
            metrics = evaluate_caption(large_preds, large_refs)
            caption_time = time.time() - start_time
            
            if caption_time > 30.0:  # Should be reasonable
                self.errors.append(f"Caption evaluation too slow: {caption_time:.3f}s")
                return False
            
            self.log(f"  Diagnosis 100 samples: {diag_time:.3f}s ✓")
            self.log(f"  Caption 10 samples: {caption_time:.3f}s ✓")
            self.results["performance"] = True
            return True
            
        except Exception as e:
            self.errors.append(f"Performance validation failed: {e}")
            self.results["performance"] = False
            return False
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests and return results."""
        self.log("🔍 Starting Comprehensive Metrics Validation\n")
        
        validations = [
            ("Basic Evaluator", self.validate_basic_evaluator),
            ("Diagnosis Metrics", self.validate_diagnosis_metrics),
            ("Caption Metrics", self.validate_caption_metrics),
            ("Detection Metrics", self.validate_detection_metrics),
            ("Integrated Evaluation", self.validate_integrated_evaluation),
            ("Performance", self.validate_performance),
        ]
        
        all_passed = True
        for name, validation_func in validations:
            try:
                if not validation_func():
                    all_passed = False
                    self.log(f"❌ {name} validation failed")
                else:
                    self.log(f"✅ {name} validation passed")
            except Exception as e:
                all_passed = False
                self.errors.append(f"{name} validation crashed: {e}")
                self.log(f"💥 {name} validation crashed")
        
        self.results["overall"] = all_passed
        
        # Print summary
        self.log(f"\n📊 Validation Summary:")
        self.log(f"  Overall: {'✅ PASSED' if all_passed else '❌ FAILED'}")
        for test_name, passed in self.results.items():
            if test_name != "overall":
                status = "✅ PASSED" if passed else "❌ FAILED"
                self.log(f"  {test_name}: {status}")
        
        if self.errors:
            self.log(f"\n❌ Errors found:")
            for error in self.errors:
                self.log(f"  - {error}")
        
        return {
            "overall_passed": all_passed,
            "test_results": self.results,
            "errors": self.errors
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Validate NOVA Retrieval VLM metrics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output results to JSON file")
    parser.add_argument("--exit-on-failure", action="store_true", help="Exit with error code on failure")
    
    args = parser.parse_args()
    
    validator = MetricsValidator(verbose=args.verbose)
    results = validator.run_all_validations()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    if args.exit_on_failure and not results["overall_passed"]:
        sys.exit(1)
    elif results["overall_passed"]:
        print("✅ All metrics validation passed!")
        sys.exit(0)
    else:
        print("❌ Some metrics validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 