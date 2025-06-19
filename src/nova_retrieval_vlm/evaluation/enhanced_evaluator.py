"""
Enhanced Evaluator with Advanced Retrieval and Visual Reasoning

This module integrates state-of-the-art retrieval techniques and radiology-specific 
visual reasoning to provide comprehensive medical VLM evaluation.

Based on insights from ChestX-Reasoner and clinical radiology practice.
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from dataclasses import dataclass, asdict
from loguru import logger

import torch
import numpy as np
from PIL import Image

from ..retrieval.advanced_retrieval import AdvancedRetriever, RetrievalResult
from ..visual_reasoning.radiology_analyzer import AdvancedRadiologyAnalyzer, RadiologyAnalysis
from ..models.model_adapter import ModelAdapter
from ..evaluation.evaluator import Evaluator, EvaluationResult

@dataclass
class EnhancedPrediction:
    """Enhanced prediction with retrieval context and visual reasoning."""
    prediction: str
    confidence: float
    retrieval_results: List[RetrievalResult]
    visual_analysis: Optional[RadiologyAnalysis]
    reasoning_chain: List[str]
    processing_time: float
    model_name: str
    task_type: str

@dataclass
class EnhancedEvaluationResult:
    """Enhanced evaluation result with detailed analysis."""
    # Base metrics
    overall_score: float
    task_scores: Dict[str, float]
    
    # Retrieval analysis
    retrieval_effectiveness: float
    retrieval_precision: float
    retrieval_recall: float
    
    # Visual reasoning analysis
    symmetry_detection_accuracy: float
    anatomical_localization_accuracy: float
    abnormality_detection_f1: float
    
    # Reasoning quality
    reasoning_completeness: float
    reasoning_accuracy: float
    clinical_relevance: float
    
    # Performance metrics
    average_processing_time: float
    confidence_calibration: float
    
    # Detailed results
    per_sample_results: List[Dict[str, Any]]
    error_analysis: Dict[str, Any]

class EnhancedEvaluator:
    """Advanced evaluator with state-of-the-art retrieval and visual reasoning."""
    
    def __init__(self, 
                 retriever: Optional[AdvancedRetriever] = None,
                 visual_analyzer: Optional[AdvancedRadiologyAnalyzer] = None,
                 base_evaluator: Optional[Evaluator] = None):
        
        self.retriever = retriever or AdvancedRetriever()
        self.visual_analyzer = visual_analyzer or AdvancedRadiologyAnalyzer()
        self.base_evaluator = base_evaluator or Evaluator()
        
        # Configure for different evaluation modes
        self.evaluation_modes = {
            'baseline': {'use_retrieval': False, 'use_visual_reasoning': False},
            'retrieval_only': {'use_retrieval': True, 'use_visual_reasoning': False},
            'visual_only': {'use_retrieval': False, 'use_visual_reasoning': True},
            'full_enhanced': {'use_retrieval': True, 'use_visual_reasoning': True}
        }
        
        # Clinical evaluation thresholds
        self.clinical_thresholds = {
            'symmetry_detection': 0.85,
            'midline_shift': 5.0,  # pixels
            'lesion_detection': 0.7,
            'confidence_threshold': 0.6
        }
    
    async def evaluate_enhanced(self,
                              model: ModelAdapter,
                              dataset: List[Dict[str, Any]],
                              mode: str = 'full_enhanced',
                              output_dir: Optional[Path] = None) -> EnhancedEvaluationResult:
        """
        Perform enhanced evaluation with retrieval and visual reasoning.
        
        Args:
            model: Model adapter for evaluation
            dataset: Dataset samples for evaluation
            mode: Evaluation mode ('baseline', 'retrieval_only', 'visual_only', 'full_enhanced')
            output_dir: Directory to save detailed results
        """
        
        logger.info(f"Starting enhanced evaluation in {mode} mode")
        start_time = time.time()
        
        # Get evaluation configuration
        config = self.evaluation_modes.get(mode, self.evaluation_modes['full_enhanced'])
        
        # Process all samples
        enhanced_predictions = []
        per_sample_results = []
        
        for i, sample in enumerate(dataset):
            logger.debug(f"Processing sample {i+1}/{len(dataset)}")
            
            # Generate enhanced prediction
            prediction = await self._generate_enhanced_prediction(
                model, sample, config
            )
            enhanced_predictions.append(prediction)
            
            # Evaluate this sample
            sample_result = await self._evaluate_sample(sample, prediction)
            per_sample_results.append(sample_result)
        
        # Aggregate results
        enhanced_result = self._aggregate_results(
            enhanced_predictions, per_sample_results, mode
        )
        
        # Save results if output directory provided
        if output_dir:
            await self._save_enhanced_results(enhanced_result, output_dir, mode)
        
        total_time = time.time() - start_time
        logger.info(f"Enhanced evaluation completed in {total_time:.2f}s")
        
        return enhanced_result
    
    async def _generate_enhanced_prediction(self,
                                          model: ModelAdapter,
                                          sample: Dict[str, Any],
                                          config: Dict[str, bool]) -> EnhancedPrediction:
        """Generate prediction with optional retrieval and visual reasoning."""
        
        start_time = time.time()
        
        # Extract basic information
        image_path = Path(sample['image_path'])
        question = sample.get('question', '')
        task_type = sample.get('task', 'general')
        
        # Step 1: Visual reasoning analysis (if enabled)
        visual_analysis = None
        if config['use_visual_reasoning']:
            try:
                visual_analysis = self.visual_analyzer.analyze_image(
                    image_path, task_context=task_type
                )
                logger.debug(f"Visual analysis completed with confidence {visual_analysis.confidence_score:.3f}")
            except Exception as e:
                logger.warning(f"Visual analysis failed: {e}")
        
        # Step 2: Retrieval augmentation (if enabled)
        retrieval_results = []
        context = ""
        if config['use_retrieval']:
            try:
                # Enhance query with visual findings if available
                enhanced_query = question
                if visual_analysis:
                    # Add visual findings to query context
                    visual_context = f" Visual findings: {visual_analysis.overall_assessment}"
                    enhanced_query += visual_context
                
                retrieval_results = self.retriever.retrieve(
                    enhanced_query, 
                    top_k=5,
                    use_query_expansion=True,
                    use_reranking=True
                )
                
                # Build context from retrieval results
                context_parts = []
                for result in retrieval_results:
                    context_parts.append(f"[{result.reasoning_type}] {result.text}")
                context = "\n".join(context_parts)
                
                logger.debug(f"Retrieved {len(retrieval_results)} relevant documents")
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
        
        # Step 3: Generate reasoning chain
        reasoning_chain = []
        if visual_analysis:
            for step in visual_analysis.reasoning_chain:
                reasoning_chain.append(f"Step {step.step_number}: {step.observation} - {step.reasoning}")
        
        # Step 4: Prepare enhanced prompt
        enhanced_prompt = self._build_enhanced_prompt(
            question, context, visual_analysis, reasoning_chain
        )
        
        # Step 5: Generate model prediction
        try:
            image = Image.open(image_path)
            prediction_result = await model.generate_async(
                prompt=enhanced_prompt,
                image=image,
                max_tokens=512
            )
            prediction = prediction_result.get('text', '')
            confidence = prediction_result.get('confidence', 0.5)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            prediction = f"Error generating prediction: {e}"
            confidence = 0.0
        
        processing_time = time.time() - start_time
        
        return EnhancedPrediction(
            prediction=prediction,
            confidence=confidence,
            retrieval_results=retrieval_results,
            visual_analysis=visual_analysis,
            reasoning_chain=reasoning_chain,
            processing_time=processing_time,
            model_name=model.model_name,
            task_type=task_type
        )
    
    def _build_enhanced_prompt(self,
                             question: str,
                             context: str,
                             visual_analysis: Optional[RadiologyAnalysis],
                             reasoning_chain: List[str]) -> str:
        """Build enhanced prompt with retrieval context and visual reasoning."""
        
        prompt_parts = []
        
        # Add system context for medical reasoning
        prompt_parts.append(
            "You are an expert radiologist analyzing medical images with access to "
            "relevant clinical guidelines and visual analysis tools. Provide systematic, "
            "step-by-step reasoning following clinical standards."
        )
        
        # Add retrieval context if available
        if context:
            prompt_parts.append(f"Relevant Clinical Guidelines:\n{context}")
        
        # Add visual analysis if available
        if visual_analysis:
            prompt_parts.append(f"Visual Analysis Summary:\n{visual_analysis.overall_assessment}")
            
            # Add symmetry analysis
            symmetry = visual_analysis.symmetry_analysis
            prompt_parts.append(
                f"Symmetry Analysis: Score={symmetry.symmetry_score:.2f}, "
                f"Midline shift={symmetry.midline_shift:.1f}px"
            )
            
            # Add detected features
            if visual_analysis.visual_features:
                features_desc = []
                for feature in visual_analysis.visual_features:
                    features_desc.append(f"{feature.name} ({feature.feature_type})")
                prompt_parts.append(f"Detected Features: {', '.join(features_desc)}")
        
        # Add reasoning chain if available
        if reasoning_chain:
            prompt_parts.append(f"Step-by-step Analysis:\n" + "\n".join(reasoning_chain))
        
        # Add the actual question
        prompt_parts.append(f"Question: {question}")
        
        # Add instruction for systematic response
        prompt_parts.append(
            "Please provide a comprehensive response that:\n"
            "1. Addresses the specific question\n"
            "2. References relevant clinical guidelines\n"
            "3. Incorporates visual findings when applicable\n"
            "4. Follows systematic radiological reasoning\n"
            "5. Indicates confidence level and any limitations"
        )
        
        return "\n\n".join(prompt_parts)
    
    async def _evaluate_sample(self,
                             sample: Dict[str, Any],
                             prediction: EnhancedPrediction) -> Dict[str, Any]:
        """Evaluate individual sample with enhanced metrics."""
        
        # Base evaluation using standard metrics
        base_result = self.base_evaluator.evaluate_prediction(
            prediction.prediction,
            sample.get('reference', ''),
            sample.get('task', 'general')
        )
        
        # Enhanced evaluation metrics
        sample_result = {
            'sample_id': sample.get('id', 'unknown'),
            'task': sample.get('task', 'general'),
            'base_score': base_result.get('score', 0.0),
            'prediction': prediction.prediction,
            'reference': sample.get('reference', ''),
            'confidence': prediction.confidence,
            'processing_time': prediction.processing_time,
        }
        
        # Retrieval evaluation
        if prediction.retrieval_results:
            sample_result.update(self._evaluate_retrieval_quality(
                prediction.retrieval_results, sample
            ))
        
        # Visual reasoning evaluation
        if prediction.visual_analysis:
            sample_result.update(self._evaluate_visual_reasoning(
                prediction.visual_analysis, sample
            ))
        
        # Reasoning chain evaluation
        if prediction.reasoning_chain:
            sample_result.update(self._evaluate_reasoning_quality(
                prediction.reasoning_chain, sample
            ))
        
        return sample_result
    
    def _evaluate_retrieval_quality(self,
                                   retrieval_results: List[RetrievalResult],
                                   sample: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate quality of retrieval results."""
        
        # Calculate retrieval metrics
        avg_confidence = np.mean([r.confidence for r in retrieval_results])
        avg_score = np.mean([r.score for r in retrieval_results])
        
        # Evaluate reasoning type diversity
        reasoning_types = set(r.reasoning_type for r in retrieval_results)
        type_diversity = len(reasoning_types) / max(len(retrieval_results), 1)
        
        # Evaluate medical concept coverage
        all_concepts = []
        for result in retrieval_results:
            all_concepts.extend(result.medical_concepts)
        concept_diversity = len(set(all_concepts)) / max(len(all_concepts), 1)
        
        # Precision / recall against ground-truth relevant documents (if provided)
        precision = recall = 0.0
        if 'relevant_docs' in sample:
            relevant_docs = set(sample['relevant_docs'])
            retrieved_docs = {r.text for r in retrieval_results}
            true_positives = len(retrieved_docs & relevant_docs)
            precision = true_positives / len(retrieved_docs) if retrieved_docs else 0.0
            recall = true_positives / len(relevant_docs) if relevant_docs else 0.0

        return {
            'retrieval_avg_confidence': avg_confidence,
            'retrieval_avg_score': avg_score,
            'retrieval_type_diversity': type_diversity,
            'retrieval_concept_diversity': concept_diversity,
            'retrieval_count': len(retrieval_results),
            'retrieval_precision': precision,
            'retrieval_recall': recall,
        }
    
    def _evaluate_visual_reasoning(self,
                                 visual_analysis: RadiologyAnalysis,
                                 sample: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate quality of visual reasoning."""
        
        metrics = {
            'visual_confidence': visual_analysis.confidence_score,
            'symmetry_score': visual_analysis.symmetry_analysis.symmetry_score,
            'midline_shift': visual_analysis.symmetry_analysis.midline_shift,
            'feature_count': len(visual_analysis.visual_features),
            'reasoning_steps': len(visual_analysis.reasoning_chain)
        }
        
        # Evaluate feature detection accuracy (if ground truth available)
        if 'visual_features' in sample:
            ground_truth_features = sample['visual_features']
            detected_features = [f.name for f in visual_analysis.visual_features]
            
            # Calculate precision and recall for feature detection
            true_positives = len(set(detected_features) & set(ground_truth_features))
            precision = true_positives / max(len(detected_features), 1)
            recall = true_positives / max(len(ground_truth_features), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            metrics.update({
                'feature_detection_precision': precision,
                'feature_detection_recall': recall,
                'feature_detection_f1': f1
            })
        
        return metrics
    
    def _evaluate_reasoning_quality(self,
                                  reasoning_chain: List[str],
                                  sample: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate quality of reasoning chain."""
        
        # Basic reasoning metrics
        reasoning_length = len(reasoning_chain)
        avg_step_length = np.mean([len(step) for step in reasoning_chain])
        
        # Assess reasoning completeness (heuristic based on content)
        completeness_indicators = [
            'symmetry', 'midline', 'ventricle', 'abnormal', 'normal',
            'assessment', 'finding', 'recommendation'
        ]
        
        completeness_score = 0.0
        for indicator in completeness_indicators:
            if any(indicator.lower() in step.lower() for step in reasoning_chain):
                completeness_score += 1.0
        
        completeness_score /= len(completeness_indicators)
        
        return {
            'reasoning_length': reasoning_length,
            'reasoning_avg_step_length': avg_step_length,
            'reasoning_completeness': completeness_score
        }
    
    def _aggregate_results(self,
                          predictions: List[EnhancedPrediction],
                          sample_results: List[Dict[str, Any]],
                          mode: str) -> EnhancedEvaluationResult:
        """Aggregate individual sample results into overall metrics."""
        
        # Basic aggregation
        overall_scores = [r.get('base_score', 0.0) for r in sample_results]
        overall_score = np.mean(overall_scores)
        
        # Task-specific scores
        task_scores = {}
        for task in set(r.get('task', 'general') for r in sample_results):
            task_results = [r['base_score'] for r in sample_results if r.get('task') == task]
            task_scores[task] = np.mean(task_results) if task_results else 0.0
        
        # Retrieval metrics
        retrieval_confidences = [r.get('retrieval_avg_confidence', 0.0) for r in sample_results]
        retrieval_effectiveness = _safe_mean([c for c in retrieval_confidences if c > 0])
        
        # Retrieval precision / recall (if available)
        retrieval_precisions = [r.get('retrieval_precision', np.nan) for r in sample_results]
        retrieval_recalls = [r.get('retrieval_recall', np.nan) for r in sample_results]
        retrieval_precision = _safe_mean(retrieval_precisions)
        retrieval_recall = _safe_mean(retrieval_recalls)
        
        # Visual reasoning metrics
        visual_confidences = [r.get('visual_confidence', 0.0) for r in sample_results]
        symmetry_scores = [r.get('symmetry_score', 0.0) for r in sample_results]
        
        symmetry_detection_accuracy = _safe_mean([s for s in symmetry_scores if s > 0])
        
        # Feature detection metrics
        feature_f1_scores = [r.get('feature_detection_f1', 0.0) for r in sample_results]
        abnormality_detection_f1 = _safe_mean([f for f in feature_f1_scores if f > 0])
        
        # Reasoning quality
        completeness_scores = [r.get('reasoning_completeness', 0.0) for r in sample_results]
        reasoning_completeness = _safe_mean([c for c in completeness_scores if c > 0])
        
        # Performance metrics
        processing_times = [p.processing_time for p in predictions]
        average_processing_time = _safe_mean(processing_times)
        
        confidences = [p.confidence for p in predictions]
        confidence_calibration = float(np.std(confidences)) if confidences else 0.0  # Lower std = better calibration
        
        # Error analysis
        error_analysis = self._analyze_errors(sample_results)
        
        return EnhancedEvaluationResult(
            overall_score=overall_score,
            task_scores=task_scores,
            retrieval_effectiveness=retrieval_effectiveness,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            symmetry_detection_accuracy=symmetry_detection_accuracy,
            anatomical_localization_accuracy=0.0,  # TODO: Implement
            abnormality_detection_f1=abnormality_detection_f1,
            reasoning_completeness=reasoning_completeness,
            reasoning_accuracy=0.0,  # TODO: Implement based on expert evaluation
            clinical_relevance=0.0,  # TODO: Implement based on expert evaluation
            average_processing_time=average_processing_time,
            confidence_calibration=confidence_calibration,
            per_sample_results=sample_results,
            error_analysis=error_analysis
        )
    
    def _analyze_errors(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common error patterns."""
        
        # Find low-scoring samples
        low_scores = [r for r in sample_results if r.get('base_score', 0.0) < 0.5]
        
        # Analyze failure modes
        error_analysis = {
            'low_score_count': len(low_scores),
            'low_score_percentage': len(low_scores) / max(len(sample_results), 1),
            'common_failure_modes': [],
            'task_specific_errors': {}
        }
        
        # Task-specific error analysis
        for task in set(r.get('task', 'general') for r in sample_results):
            task_errors = [r for r in low_scores if r.get('task') == task]
            error_analysis['task_specific_errors'][task] = {
                'count': len(task_errors),
                'percentage': len(task_errors) / max(len([r for r in sample_results if r.get('task') == task]), 1)
            }
        
        return error_analysis
    
    async def _save_enhanced_results(self,
                                   result: EnhancedEvaluationResult,
                                   output_dir: Path,
                                   mode: str) -> None:
        """Save enhanced evaluation results to files."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        main_results = {
            'mode': mode,
            'overall_score': result.overall_score,
            'task_scores': result.task_scores,
            'retrieval_metrics': {
                'effectiveness': result.retrieval_effectiveness,
                'precision': result.retrieval_precision,
                'recall': result.retrieval_recall
            },
            'visual_reasoning_metrics': {
                'symmetry_detection_accuracy': result.symmetry_detection_accuracy,
                'anatomical_localization_accuracy': result.anatomical_localization_accuracy,
                'abnormality_detection_f1': result.abnormality_detection_f1
            },
            'reasoning_quality': {
                'completeness': result.reasoning_completeness,
                'accuracy': result.reasoning_accuracy,
                'clinical_relevance': result.clinical_relevance
            },
            'performance': {
                'average_processing_time': result.average_processing_time,
                'confidence_calibration': result.confidence_calibration
            },
            'error_analysis': result.error_analysis
        }
        
        with open(output_dir / f"enhanced_results_{mode}.json", 'w') as f:
            json.dump(main_results, f, indent=2)
        
        # Save detailed per-sample results
        with open(output_dir / f"per_sample_results_{mode}.json", 'w') as f:
            json.dump(result.per_sample_results, f, indent=2)
        
        logger.info(f"Enhanced results saved to {output_dir}")

# Convenience function for running enhanced evaluation
async def run_enhanced_evaluation(model: ModelAdapter,
                                dataset: List[Dict[str, Any]],
                                retriever_path: Optional[Path] = None,
                                output_dir: Optional[Path] = None,
                                mode: str = 'full_enhanced') -> EnhancedEvaluationResult:
    """
    Run enhanced evaluation with state-of-the-art capabilities.
    
    Args:
        model: Model adapter for evaluation
        dataset: Dataset samples for evaluation
        retriever_path: Path to pre-built retriever index
        output_dir: Directory to save results
        mode: Evaluation mode
    """
    
    # Initialize components
    retriever = AdvancedRetriever()
    if retriever_path and retriever_path.exists():
        retriever.load_index(retriever_path)
    
    visual_analyzer = AdvancedRadiologyAnalyzer()
    evaluator = EnhancedEvaluator(retriever, visual_analyzer)
    
    # Run evaluation
    return await evaluator.evaluate_enhanced(
        model, dataset, mode, output_dir
    )

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_mean(values: list[float]) -> float:
    """Compute the mean of *values* safely.

    - Filters out *None* and *NaN* values.
    - Returns ``0.0`` when the resulting list is empty.
    """
    import numpy as _np  # local import to avoid polluting public namespace

    filtered = [v for v in values if v is not None and not _np.isnan(v)]
    return float(_np.mean(filtered)) if filtered else 0.0 