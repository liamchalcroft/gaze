"""
Visual Reasoning for Radiology

Implements visual analysis techniques for medical imaging based on structured
reasoning approaches from clinical radiology practice.

Provides symmetry analysis, anatomical structure detection, anomaly detection,
and step-by-step reasoning chain generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from nova_retrieval_vlm.types import ImageArray
from nova_retrieval_vlm.types import tensor_validated

# Constants
MIN_DIMENSIONS_FOR_GRAYSCALE = 2

# Analysis constants
MINIMUM_VENTRICLE_AREA = 100
LESION_SIZE_RANGE_MIN = 50
LESION_SIZE_RANGE_MAX = 5000
ASYMMETRY_THRESHOLD = 0.15
REGION_ANALYSIS_SIZE = 64
SYMMETRY_CONCERN_THRESHOLD = 0.85
MIDLINE_SHIFT_CONCERN_PIXELS = 5.0

# Visual processing constants
RGB_CHANNELS = 3
MIN_REGION_WIDTH = 10
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0
MIN_SOLIDITY = 0.3
CONFIDENCE_THRESHOLD = 0.6
DEFAULT_CONFIDENCE = 0.5
HIGH_CONFIDENCE = 0.8
LESION_CONCERN_SIZE = 1000


@dataclass
class VisualFeature:
    """Represents a visual feature detected in medical images."""

    name: str
    location: tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    description: str
    anatomical_region: str
    feature_type: str  # 'normal', 'abnormal', 'uncertain'


@dataclass
class SymmetryAnalysis:
    """Results of bilateral symmetry analysis."""

    symmetry_score: float  # 0-1, higher = more symmetric
    asymmetry_regions: list[dict[str, Any]]
    midline_shift: float  # pixels of deviation from expected midline
    laterality_findings: dict[str, Any]


@dataclass
class ReasoningStep:
    """Individual step in medical reasoning chain."""

    step_number: int
    observation: str
    reasoning: str
    confidence: float
    anatomical_focus: str
    supporting_evidence: list[str]


@dataclass
class RadiologyAnalysis:
    """Complete radiology analysis result."""

    visual_features: list[VisualFeature]
    symmetry_analysis: SymmetryAnalysis
    reasoning_chain: list[ReasoningStep]
    overall_assessment: str
    confidence_score: float
    recommended_followup: list[str]


class BrainStructureDetector:
    """Detect and localize brain structures in MRI images."""

    def __init__(self):
        # Define brain structure templates and characteristics
        self.brain_structures = {
            "ventricles": {
                "characteristics": ["dark", "bilateral", "symmetric", "central"],
                "expected_location": "central",
                "normal_size_range": (0.02, 0.08),  # as fraction of brain area
            },
            "cerebellum": {
                "characteristics": ["posterior", "bilateral", "foliated"],
                "expected_location": "posterior_inferior",
                "normal_size_range": (0.10, 0.15),
            },
            "brainstem": {
                "characteristics": ["central", "vertical", "continuous"],
                "expected_location": "central_posterior",
                "normal_size_range": (0.03, 0.06),
            },
            "cortex": {
                "characteristics": ["peripheral", "gray_matter", "folded"],
                "expected_location": "peripheral",
                "normal_size_range": (0.30, 0.45),
            },
        }

    @tensor_validated
    def detect_structures(self, image: ImageArray) -> list[VisualFeature]:
        """Detect anatomical structures in brain MRI with tensor validation."""
        features = []

        # Convert to grayscale if needed with proper tensor validation
        if len(image.shape) == RGB_CHANNELS:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.squeeze() if len(image.shape) > MIN_DIMENSIONS_FOR_GRAYSCALE else image

        # Ensure proper uint8 format for OpenCV operations
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        # Normalize image
        gray = cv2.normalize(gray, np.zeros_like(gray), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = npt.NDArray[np.uint8](gray)

        # Detect ventricles (dark regions in central area)
        ventricle_features = self._detect_ventricles(gray)
        features.extend(ventricle_features)

        # Detect brain boundary
        brain_boundary = self._detect_brain_boundary(gray)
        if brain_boundary:
            features.append(brain_boundary)

        # Detect potential lesions (abnormal intensities)
        lesion_features = self._detect_potential_lesions(gray)
        features.extend(lesion_features)

        return features

    def _detect_ventricles(self, image: npt.NDArray[np.uint8]) -> list[VisualFeature]:
        """Detect ventricular system."""
        features = []
        h, w = image.shape

        # Focus on central region where ventricles are expected
        central_region = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

        # Apply threshold to find dark regions (CSF)
        _, binary = cv2.threshold(central_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary  # Invert so dark regions are white

        # Find connected components
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MINIMUM_VENTRICLE_AREA:
                x, y, width, height = cv2.boundingRect(contour)
                # Adjust coordinates to full image
                x += w // 4
                y += h // 4

                # Calculate features
                aspect_ratio = width / height if height > 0 else 0
                solidity = area / (width * height) if width * height > 0 else 0

                # Assess if this looks like ventricles
                confidence = DEFAULT_CONFIDENCE
                if MIN_ASPECT_RATIO < aspect_ratio < MAX_ASPECT_RATIO and solidity > MIN_SOLIDITY:
                    confidence = HIGH_CONFIDENCE

                feature = VisualFeature(
                    name="ventricles",
                    location=(x, y, width, height),
                    confidence=confidence,
                    description=f"Dark CSF-filled region, aspect ratio: {aspect_ratio:.2f}",
                    anatomical_region="central",
                    feature_type="normal" if confidence > CONFIDENCE_THRESHOLD else "uncertain",
                )
                features.append(feature)

        return features

    def _detect_brain_boundary(self, image: npt.NDArray[np.uint8]) -> VisualFeature | None:
        """Detect overall brain boundary."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Threshold to separate brain from background
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find largest contour (should be brain boundary)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            return VisualFeature(
                name="brain_boundary",
                location=(x, y, w, h),
                confidence=0.9,
                description="Overall brain parenchyma boundary",
                anatomical_region="whole_brain",
                feature_type="normal",
            )

        return None

    def _detect_potential_lesions(self, image: npt.NDArray[np.uint8]) -> list[VisualFeature]:
        """Detect potential abnormal regions using intensity analysis."""
        features = []

        # Calculate image statistics
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)

        # Create binary mask for extreme intensities
        high_intensity_mask = image > (mean_intensity + 2 * std_intensity)

        # Process high intensity regions
        high_contours, _ = cv2.findContours(
            high_intensity_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in high_contours:
            area = cv2.contourArea(contour)
            if LESION_SIZE_RANGE_MIN < area < LESION_SIZE_RANGE_MAX:
                x, y, w, h = cv2.boundingRect(contour)

                feature = VisualFeature(
                    name="hyperintense_region",
                    location=(x, y, w, h),
                    confidence=0.6,
                    description=f"Hyperintense region, area: {area} pixels",
                    anatomical_region="to_be_determined",
                    feature_type="abnormal",
                )
                features.append(feature)

        return features


class BilateralSymmetryAnalyzer:
    """Analyze bilateral symmetry in brain MRI images."""

    def __init__(self):
        self.symmetry_threshold = SYMMETRY_CONCERN_THRESHOLD

    @tensor_validated
    def analyze_symmetry(self, image: ImageArray) -> SymmetryAnalysis:
        """Perform comprehensive symmetry analysis with tensor validation."""

        # Convert to grayscale with proper tensor handling
        if len(image.shape) == RGB_CHANNELS:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.squeeze() if len(image.shape) > MIN_DIMENSIONS_FOR_GRAYSCALE else image

        # Ensure proper data type for OpenCV operations
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

        # Find anatomical midline
        midline_x = self._detect_midline(gray)
        expected_midline = gray.shape[1] // 2
        midline_shift = abs(midline_x - expected_midline)

        # Calculate symmetry score
        symmetry_score = self._calculate_symmetry_score(gray, midline_x)

        # Identify asymmetric regions
        asymmetry_regions = self._find_asymmetric_regions(gray, midline_x)

        # Analyze laterality
        laterality_findings = self._analyze_laterality(gray, midline_x, asymmetry_regions)

        return SymmetryAnalysis(
            symmetry_score=symmetry_score,
            asymmetry_regions=asymmetry_regions,
            midline_shift=midline_shift,
            laterality_findings=laterality_findings,
        )

    def _detect_midline(self, image: npt.NDArray[np.uint8]) -> int:
        """Detect anatomical midline using various methods."""
        h, w = image.shape

        # Method 1: Intensity-based midline detection
        # Sum intensities along vertical strips

        # Find the line of symmetry by comparing left and right sides
        optimal_midline = w // 2
        lowest_asymmetry_score = float("inf")

        search_range = min(w // 4, 50)  # Search within reasonable range

        for candidate_midline in range(w // 2 - search_range, w // 2 + search_range):
            if candidate_midline <= 0 or candidate_midline >= w - 1:
                continue

            # Calculate asymmetry score for this candidate midline
            left_side = image[:, :candidate_midline]
            right_side = image[:, candidate_midline:]

            # Flip right side and resize to match left side
            right_flipped = np.fliplr(right_side)
            min_width = min(left_side.shape[1], right_flipped.shape[1])

            if min_width > MIN_REGION_WIDTH:  # Ensure sufficient width for comparison
                left_crop = left_side[:, -min_width:]
                right_crop = right_flipped[:, :min_width]

                # Calculate difference
                diff = np.mean(np.abs(left_crop.astype(float) - right_crop.astype(float)))

                if diff < lowest_asymmetry_score:
                    lowest_asymmetry_score = diff
                    optimal_midline = candidate_midline

        return optimal_midline

    def _calculate_symmetry_score(self, image: npt.NDArray[np.uint8], midline_x: int) -> float:
        """Calculate overall symmetry score (0-1, higher = more symmetric)."""
        h, w = image.shape

        # Extract left and right halves
        left_half = image[:, :midline_x]
        right_half = image[:, midline_x:]

        # Flip right half
        right_flipped = np.fliplr(right_half)

        # Resize to same dimensions
        min_width = min(left_half.shape[1], right_flipped.shape[1])
        if min_width <= 0:
            return 0.0

        left_resized = left_half[:, -min_width:]
        right_resized = right_flipped[:, :min_width]

        # Calculate normalized correlation
        correlation = cv2.matchTemplate(left_resized, right_resized, cv2.TM_CCOEFF_NORMED)
        symmetry_score = np.max(correlation)

        # Ensure score is between 0 and 1
        symmetry_score = max(0.0, min(1.0, (symmetry_score + 1) / 2))

        return symmetry_score

    def _find_asymmetric_regions(
        self, image: npt.NDArray[np.uint8], midline_x: int
    ) -> list[dict[str, Any]]:
        """Identify specific regions showing asymmetry."""
        asymmetric_regions = []
        h, w = image.shape

        # Divide image into regions for analysis
        region_size = REGION_ANALYSIS_SIZE

        for y in range(0, h - region_size, region_size // 2):
            for x in range(0, midline_x - region_size, region_size // 2):
                # Get left region
                left_region = image[y : y + region_size, x : x + region_size]

                # Get corresponding right region (mirrored)
                right_x = 2 * midline_x - x - region_size
                if right_x >= 0 and right_x + region_size < w:
                    right_region = image[y : y + region_size, right_x : right_x + region_size]
                    right_region = np.fliplr(right_region)

                    # Calculate local asymmetry
                    diff = np.mean(np.abs(left_region.astype(float) - right_region.astype(float)))
                    normalized_diff = diff / 255.0  # Normalize to 0-1

                    if normalized_diff > ASYMMETRY_THRESHOLD:
                        asymmetric_regions.append(
                            {
                                "location": (x, y, region_size, region_size),
                                "asymmetry_score": normalized_diff,
                                "side": "left",
                                "description": (
                                    f"Asymmetric region with score {normalized_diff:.3f}"
                                ),
                            }
                        )

        return asymmetric_regions

    def _analyze_laterality(
        self, image: npt.NDArray[np.uint8], midline_x: int, asymmetry_regions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze lateralized findings."""
        h, w = image.shape

        left_half = image[:, :midline_x]
        right_half = image[:, midline_x:]

        # Calculate mean intensities
        left_mean = np.mean(left_half)
        right_mean = np.mean(right_half)

        # Calculate volume differences
        left_volume = np.sum(left_half > np.mean(image))
        right_volume = np.sum(right_half > np.mean(image))

        laterality_findings = {
            "left_mean_intensity": float(left_mean),
            "right_mean_intensity": float(right_mean),
            "intensity_asymmetry": float(abs(left_mean - right_mean)),
            "left_volume": int(left_volume),
            "right_volume": int(right_volume),
            "volume_asymmetry": float(
                abs(left_volume - right_volume) / max(left_volume, right_volume, 1)
            ),
            "asymmetric_region_count": len(asymmetry_regions),
            "predominant_side": "left"
            if left_mean > right_mean
            else "right"
            if right_mean > left_mean
            else "symmetric",
        }

        return laterality_findings


class MedicalReasoningChain:
    """Generate step-by-step reasoning chains for medical imaging analysis."""

    def __init__(self):
        self.reasoning_templates = {
            "symmetry_analysis": [
                "Assess bilateral symmetry of brain structures",
                "Identify midline position and any shift",
                "Compare left and right hemispheres for volume and intensity",
                "Evaluate ventricular symmetry and size",
            ],
            "lesion_detection": [
                "Survey brain parenchyma for abnormal signal intensities",
                "Assess lesion characteristics: size, shape, location",
                "Evaluate mass effect and surrounding edema",
                "Consider differential diagnosis based on imaging features",
            ],
            "vascular_assessment": [
                "Examine for signs of acute hemorrhage",
                "Assess vascular territories for ischemic changes",
                "Evaluate for midline shift or herniation",
                "Check for hydrocephalus or ventricular enlargement",
            ],
        }

    def generate_reasoning_chain(
        self,
        visual_features: list[VisualFeature],
        symmetry_analysis: SymmetryAnalysis,
        task_type: str = "general_assessment",  # noqa: ARG002 - Reserved for future task-specific reasoning
    ) -> list[ReasoningStep]:
        """Generate step-by-step reasoning based on findings."""

        reasoning_steps = []
        step_number = 1

        # Step 1: Overall image quality and orientation
        reasoning_steps.append(
            ReasoningStep(
                step_number=step_number,
                observation="Brain MRI image acquired in axial plane",
                reasoning=(
                    "Standard radiological assessment begins with image quality "
                    "and orientation verification"
                ),
                confidence=0.95,
                anatomical_focus="whole_brain",
                supporting_evidence=["Image orientation markers", "Anatomical landmarks visible"],
            )
        )
        step_number += 1

        # Step 2: Symmetry assessment
        if symmetry_analysis.symmetry_score < SYMMETRY_CONCERN_THRESHOLD:
            reasoning_steps.append(
                ReasoningStep(
                    step_number=step_number,
                    observation=(
                        f"Bilateral asymmetry detected "
                        f"(symmetry score: {symmetry_analysis.symmetry_score:.2f})"
                    ),
                    reasoning=(
                        "Asymmetry may indicate pathological process "
                        "affecting one hemisphere more than the other"
                    ),
                    confidence=0.8,
                    anatomical_focus="bilateral_hemispheres",
                    supporting_evidence=[
                        f"Midline shift: {symmetry_analysis.midline_shift:.1f} pixels",
                        f"Asymmetric regions: {len(symmetry_analysis.asymmetry_regions)}",
                    ],
                )
            )
            step_number += 1

        # Step 3: Ventricular assessment
        ventricle_features = [f for f in visual_features if f.name == "ventricles"]
        if ventricle_features:
            reasoning_steps.append(
                ReasoningStep(
                    step_number=step_number,
                    observation=(
                        f"Ventricular system identified with {len(ventricle_features)} components"
                    ),
                    reasoning=(
                        "Ventricular size and symmetry are key indicators of "
                        "intracranial pressure and structural abnormalities"
                    ),
                    confidence=0.85,
                    anatomical_focus="ventricular_system",
                    supporting_evidence=[
                        f"Ventricle locations: {[f.location for f in ventricle_features]}"
                    ],
                )
            )
            step_number += 1

        # Step 4: Abnormal findings
        abnormal_features = [f for f in visual_features if f.feature_type == "abnormal"]
        if abnormal_features:
            for feature in abnormal_features:
                reasoning_steps.append(
                    ReasoningStep(
                        step_number=step_number,
                        observation=f"Abnormal finding detected: {feature.name}",
                        reasoning=(
                            f"Located in {feature.anatomical_region}, requires further evaluation"
                        ),
                        confidence=feature.confidence,
                        anatomical_focus=feature.anatomical_region,
                        supporting_evidence=[feature.description, f"Location: {feature.location}"],
                    )
                )
                step_number += 1

        # Step 5: Overall assessment
        overall_confidence = np.mean([step.confidence for step in reasoning_steps])
        reasoning_steps.append(
            ReasoningStep(
                step_number=step_number,
                observation="Systematic review completed",
                reasoning="Integration of all findings to formulate radiological impression",
                confidence=overall_confidence,
                anatomical_focus="whole_brain",
                supporting_evidence=[
                    "Symmetry analysis",
                    "Structural assessment",
                    "Abnormality detection",
                ],
            )
        )

        return reasoning_steps


class MedicalImageAnalyzer:
    """Integrates brain structure detection, symmetry analysis and medical reasoning."""

    def __init__(self):
        self.structure_detector = BrainStructureDetector()
        self.symmetry_analyzer = BilateralSymmetryAnalyzer()
        self.reasoning_chain = MedicalReasoningChain()

        # Clinical decision thresholds
        self.thresholds = {
            "symmetry_concern": SYMMETRY_CONCERN_THRESHOLD,
            "midline_shift_concern": MIDLINE_SHIFT_CONCERN_PIXELS,
            "lesion_size_concern": LESION_CONCERN_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }

    def analyze_image(self, image_path: Path, task_context: str = "general") -> RadiologyAnalysis:
        """
        Perform comprehensive radiology analysis of brain MRI.

        Args:
            image_path: Path to brain MRI image
            task_context: Clinical context ('localization', 'diagnosis', 'screening')
        """

        logger.info(f"Starting radiology analysis of {image_path}")

        # Load and preprocess image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Step 1: Detect anatomical structures and features
        visual_features = self.structure_detector.detect_structures(image)
        logger.debug(f"Detected {len(visual_features)} visual features")

        # Step 2: Analyze symmetry
        symmetry_analysis = self.symmetry_analyzer.analyze_symmetry(image)
        logger.debug(f"Symmetry score: {symmetry_analysis.symmetry_score:.3f}")

        # Step 3: Generate reasoning chain
        reasoning_chain = self.reasoning_chain.generate_reasoning_chain(
            visual_features, symmetry_analysis, task_context
        )

        # Step 4: Formulate overall assessment
        overall_assessment = self._formulate_assessment(
            visual_features, symmetry_analysis, reasoning_chain
        )

        # Step 5: Calculate confidence score
        confidence_score = self._calculate_overall_confidence(
            visual_features, symmetry_analysis, reasoning_chain
        )

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            visual_features, symmetry_analysis, confidence_score
        )

        analysis = RadiologyAnalysis(
            visual_features=visual_features,
            symmetry_analysis=symmetry_analysis,
            reasoning_chain=reasoning_chain,
            overall_assessment=overall_assessment,
            confidence_score=confidence_score,
            recommended_followup=recommendations,
        )

        logger.info(f"Analysis completed with confidence {confidence_score:.3f}")
        return analysis

    def _formulate_assessment(
        self,
        visual_features: list[VisualFeature],
        symmetry_analysis: SymmetryAnalysis,
        reasoning_chain: list[ReasoningStep],  # noqa: ARG002 - Reserved for future integration
    ) -> str:
        """Formulate overall radiological assessment."""

        assessment_parts = []

        # Symmetry assessment
        if symmetry_analysis.symmetry_score < self.thresholds["symmetry_concern"]:
            assessment_parts.append(
                f"Bilateral asymmetry noted (score: {symmetry_analysis.symmetry_score:.2f})"
            )
        else:
            assessment_parts.append("Brain demonstrates normal bilateral symmetry")

        # Midline assessment
        if symmetry_analysis.midline_shift > self.thresholds["midline_shift_concern"]:
            assessment_parts.append(
                f"Midline shift of {symmetry_analysis.midline_shift:.1f} pixels detected"
            )

        # Structural findings
        abnormal_features = [f for f in visual_features if f.feature_type == "abnormal"]
        if abnormal_features:
            assessment_parts.append(f"{len(abnormal_features)} abnormal finding(s) identified")
        else:
            assessment_parts.append("No obvious structural abnormalities detected")

        # Ventricular assessment
        ventricle_features = [f for f in visual_features if f.name == "ventricles"]
        if ventricle_features:
            assessment_parts.append(
                f"Ventricular system visualized ({len(ventricle_features)} components)"
            )

        return ". ".join(assessment_parts) + "."

    def _calculate_overall_confidence(
        self,
        visual_features: list[VisualFeature],
        symmetry_analysis: SymmetryAnalysis,
        reasoning_chain: list[ReasoningStep],
    ) -> float:
        """Calculate overall confidence in the analysis."""

        confidence_factors = []

        # Feature detection confidence
        if visual_features:
            feature_confidences = [f.confidence for f in visual_features]
            confidence_factors.append(np.mean(feature_confidences))

        # Symmetry analysis confidence (based on score clarity)
        symmetry_confidence = 1.0 - abs(symmetry_analysis.symmetry_score - 0.5) * 2
        confidence_factors.append(symmetry_confidence)

        # Reasoning chain confidence
        if reasoning_chain:
            reasoning_confidences = [step.confidence for step in reasoning_chain]
            confidence_factors.append(np.mean(reasoning_confidences))

        # Calculate weighted average
        overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5

        return float(overall_confidence)

    def _generate_recommendations(
        self,
        visual_features: list[VisualFeature],
        symmetry_analysis: SymmetryAnalysis,
        confidence_score: float,
    ) -> list[str]:
        """Generate clinical recommendations based on findings."""

        recommendations = []

        # Low confidence recommendations
        if confidence_score < self.thresholds["confidence_threshold"]:
            recommendations.append("Consider repeat imaging with optimized technique")
            recommendations.append("Clinical correlation recommended")

        # Asymmetry recommendations
        if symmetry_analysis.symmetry_score < self.thresholds["symmetry_concern"]:
            recommendations.append("Consider contrast-enhanced imaging")
            recommendations.append("Neurological examination correlation advised")

        # Midline shift recommendations
        if symmetry_analysis.midline_shift > self.thresholds["midline_shift_concern"]:
            recommendations.append("Urgent neurosurgical consultation")
            recommendations.append("Monitor for signs of increased intracranial pressure")

        # Abnormal findings recommendations
        abnormal_features = [f for f in visual_features if f.feature_type == "abnormal"]
        if abnormal_features:
            recommendations.append("Further characterization with additional sequences")
            recommendations.append("Consider multidisciplinary team review")

        # Default recommendation if nothing specific
        if not recommendations:
            recommendations.append("No immediate follow-up required based on current findings")

        return recommendations
