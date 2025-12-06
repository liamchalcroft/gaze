"""NOVA benchmark metric types.

Provides Pydantic models for NOVA task evaluation metrics.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field


class EvaluationMetrics(BaseModel):
    """Base evaluation metrics."""

    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float | None = Field(None, ge=0.0, le=1.0)
    recall: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)
    auc_roc: float | None = Field(None, ge=0.0, le=1.0)


class CaptionMetrics(BaseModel):
    """Evaluation metrics for caption generation tasks.

    All metrics are normalized to 0-1 range.
    """

    bleu: float = Field(ge=0.0, le=1.0, description="BLEU score (primary metric)")
    bert_f1: float = Field(ge=0.0, le=1.0, description="BERTScore F1")
    meteor: float = Field(ge=0.0, le=1.0, description="METEOR score")
    modality_f1: float = Field(ge=0.0, le=1.0, description="Modality keyword F1")
    clinical_f1: float = Field(ge=0.0, le=1.0, description="Clinical keyword F1")
    binary_accuracy: float = Field(ge=0.0, le=1.0, description="Normal/abnormal accuracy")
    binary_f1: float = Field(ge=0.0, le=1.0, description="Normal/abnormal F1")
    radgraph_f1: float | None = Field(
        None, ge=0.0, le=1.0, description="RadGraph F1 (optional dependency)"
    )


class DetectionMetrics(BaseModel):
    """Evaluation metrics for object detection/localization tasks.

    Follows NOVA benchmark protocol with multiple IoU thresholds.
    """

    map30: float = Field(ge=0.0, le=1.0, description="mAP at IoU threshold 0.3")
    map50: float = Field(ge=0.0, le=1.0, description="mAP at IoU threshold 0.5 (primary)")
    map50_95: float = Field(ge=0.0, le=1.0, description="mAP averaged across IoU 0.5-0.95")
    acc50: float = Field(ge=0.0, le=1.0, description="Detection accuracy at IoU 0.5")
    tp30: int = Field(ge=0, description="True positives at IoU 0.3")
    fp30: int = Field(ge=0, description="False positives at IoU 0.3")


class DiagnosisMetrics(BaseModel):
    """Evaluation metrics for diagnosis classification tasks.

    Follows NOVA benchmark protocol with LLM semantic matching.
    """

    top1: float = Field(ge=0.0, le=1.0, description="Top-1 accuracy (primary)")
    top5: float = Field(ge=0.0, le=1.0, description="Top-5 accuracy")
    coverage: float = Field(ge=0.0, description="Unique predictions / unique references")
    entropy: float = Field(ge=0.0, description="Shannon entropy of prediction distribution")
