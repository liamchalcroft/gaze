"""Location reference verification reward for GEMeX-ThinkVG.

Computes reward based on anatomical region matching between
predicted and ground truth location references.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from beartype import beartype

# Anatomical region synonyms and hierarchies
ANATOMICAL_SYNONYMS: dict[str, set[str]] = {
    # Lung regions
    "bilateral lung": {"both lungs", "bilateral lungs", "lungs bilaterally"},
    "right lung": {"right pulmonary", "r lung"},
    "left lung": {"left pulmonary", "l lung"},
    "right upper lobe": {
        "rul",
        "right upper zone",
        "right upper field",
        "right apex",
        "right apical",
    },
    "right middle lobe": {"rml", "right mid zone", "right middle field", "right mid field"},
    "right lower lobe": {
        "rll",
        "right lower zone",
        "right lower field",
        "right base",
        "right basilar",
    },
    "left upper lobe": {"lul", "left upper zone", "left upper field", "left apex", "left apical"},
    "left lower lobe": {"lll", "left lower zone", "left lower field", "left base", "left basilar"},
    "lingula": {"left middle lobe"},
    # Hilum
    "right hilum": {"right hilar", "right hilar region"},
    "left hilum": {"left hilar", "left hilar region"},
    "bilateral hilum": {"bilateral hilar", "perihilar", "parahilar", "hilar"},
    # Mediastinum
    "mediastinum": {"mediastinal", "cardiomediastinal", "retrosternal"},
    "heart": {"cardiac", "cardiac silhouette", "heart shadow"},
    "aorta": {"aortic", "aortic arch", "aortic knob"},
    "trachea": {"tracheal", "paratracheal"},
    # Pleural
    "pleura": {"pleural", "pleural space"},
    "costophrenic angle": {"cp angle", "costophrenic recess"},
    "right costophrenic angle": {"right cp angle"},
    "left costophrenic angle": {"left cp angle"},
    # Bony structures
    "ribs": {"rib", "costal"},
    "spine": {"spinal", "vertebral", "vertebrae"},
    "clavicle": {"clavicular"},
    # Diaphragm
    "diaphragm": {"diaphragmatic", "hemidiaphragm"},
    "right hemidiaphragm": {"right diaphragm"},
    "left hemidiaphragm": {"left diaphragm"},
    # Retrocardiac
    "retrocardiac": {"retrocardiac region", "behind heart"},
    # General lung apex region
    "lung apex": {"apical", "apex", "apices"},
    # Lung base — standalone "basilar" without laterality
    "lung base": {"basilar", "basal", "bases"},
    # Peribronchial / bronchial regions
    "peribronchial": {"peribronchial region", "bronchial"},
    # Subcarinal (mediastinal lymph node station)
    "subcarinal": {"subcarinal region", "subcarinal space"},
    # Anterior / posterior mediastinum subdivisions
    "anterior mediastinum": {"anterior mediastinal", "prevascular"},
    "posterior mediastinum": {"posterior mediastinal", "paravertebral"},
    # Whole chest (top-level hierarchy node)
    "chest": {"thorax", "thoracic", "chest wall"},
}

# Region hierarchy for partial matching
REGION_HIERARCHY: dict[str, list[str]] = {
    "bilateral lung": ["right lung", "left lung", "lung apex", "lung base", "peribronchial"],
    "right lung": ["right upper lobe", "right middle lobe", "right lower lobe", "right hilum"],
    "left lung": ["left upper lobe", "lingula", "left lower lobe", "left hilum"],
    "bilateral hilum": ["right hilum", "left hilum"],
    "mediastinum": [
        "heart",
        "aorta",
        "trachea",
        "retrocardiac",
        "subcarinal",
        "anterior mediastinum",
        "posterior mediastinum",
    ],
    "pleura": ["costophrenic angle", "right costophrenic angle", "left costophrenic angle"],
    "diaphragm": ["right hemidiaphragm", "left hemidiaphragm"],
    "chest": [
        "bilateral lung",
        "mediastinum",
        "pleura",
        "diaphragm",
        "bilateral hilum",
    ],
}


@beartype
def normalize_location(location: str) -> str:
    """Normalize anatomical location string.

    Args:
        location: Raw location string

    Returns:
        Normalized location
    """
    location = location.lower().strip()

    # Remove common filler words
    filler_words = ["the", "a", "an", "of", "in", "at", "on", "within"]
    for word in filler_words:
        location = re.sub(rf"\b{word}\b", " ", location)

    # Collapse whitespace
    location = " ".join(location.split())

    return location


@beartype
def get_canonical_region(location: str) -> str | None:
    """Get canonical region name from location string.

    Matching precedence (highest first):
    1. Exact match on canonical name
    2. Exact match on a synonym
    3. Substring match — scored by the length of the matched term so
       that more-specific regions win (e.g. "right lower lobe" beats
       "right lung" when the input contains both substrings).

    Args:
        location: Normalized location string

    Returns:
        Canonical region name or None if not found
    """
    location = normalize_location(location)

    # 1. Direct match on canonical key
    if location in ANATOMICAL_SYNONYMS:
        return location

    # 2. Exact synonym match
    for canonical, synonyms in ANATOMICAL_SYNONYMS.items():
        if location in synonyms:
            return canonical

    # 3. Substring match — collect all candidates, prefer longest match.
    #    Require the matched term to be at least MIN_SUBSTR_LEN characters
    #    to prevent very short inputs ("left", "right") from over-matching.
    MIN_SUBSTR_LEN = 4
    best_canonical: str | None = None
    best_match_len = 0

    # Skip substring fallback entirely for very short inputs — they are
    # too ambiguous to map reliably.
    if len(location) < MIN_SUBSTR_LEN:
        return None

    for canonical, synonyms in ANATOMICAL_SYNONYMS.items():
        # Check canonical as substring (either direction)
        if canonical in location or location in canonical:
            match_len = len(canonical)
            if match_len >= MIN_SUBSTR_LEN and match_len > best_match_len:
                best_match_len = match_len
                best_canonical = canonical

        # Check each synonym as substring
        for syn in synonyms:
            if syn in location or location in syn:
                match_len = len(syn)
                if match_len >= MIN_SUBSTR_LEN and match_len > best_match_len:
                    best_match_len = match_len
                    best_canonical = canonical

    return best_canonical


@beartype
def compute_region_match(pred: str, ref: str) -> float:
    """Compute exact region match score.

    Args:
        pred: Predicted location (normalized)
        ref: Reference location (normalized)

    Returns:
        1.0 for exact match, 0.0 otherwise
    """
    pred_canonical = get_canonical_region(pred)
    ref_canonical = get_canonical_region(ref)

    if pred_canonical is None or ref_canonical is None:
        # Fall back to string matching
        return 1.0 if pred == ref else 0.0

    return 1.0 if pred_canonical == ref_canonical else 0.0


def _is_ancestor(ancestor: str, descendant: str) -> int | None:
    """Return the depth if *ancestor* is a (transitive) parent of *descendant*.

    Depth 1 = direct parent, depth 2 = grandparent, etc.
    Returns None if there is no ancestor relationship.
    """
    # BFS from ancestor downward
    frontier = [(ancestor, 0)]
    visited: set[str] = set()
    while frontier:
        node, depth = frontier.pop(0)
        if node in visited:
            continue
        visited.add(node)
        children = REGION_HIERARCHY.get(node, [])
        for child in children:
            if child == descendant:
                return depth + 1
            frontier.append((child, depth + 1))
    return None


@beartype
def compute_hierarchy_match(pred: str, ref: str) -> float:
    """Compute hierarchical region match score.

    Gives partial credit when prediction and reference are related
    through the anatomical hierarchy, including transitive (grandparent)
    relationships.  Closer relationships score higher.

    Args:
        pred: Predicted location
        ref: Reference location

    Returns:
        Score in [0, 1] based on hierarchical relationship
    """
    pred_canonical = get_canonical_region(pred)
    ref_canonical = get_canonical_region(ref)

    if pred_canonical is None or ref_canonical is None:
        return 0.0

    # Exact match
    if pred_canonical == ref_canonical:
        return 1.0

    # Pred is ancestor of ref (prediction too general).
    # Deeper distance → less credit: 0.5 for depth-1, 0.35 for depth-2, …
    depth = _is_ancestor(pred_canonical, ref_canonical)
    if depth is not None:
        return max(0.1, 0.5 / depth)

    # Ref is ancestor of pred (prediction more specific, usually fine).
    depth = _is_ancestor(ref_canonical, pred_canonical)
    if depth is not None:
        return max(0.2, 0.7 / depth)

    # Check if they share a common parent (siblings)
    for children in REGION_HIERARCHY.values():
        if pred_canonical in children and ref_canonical in children:
            return 0.3  # Same general area

    return 0.0


@beartype
def compute_token_match(pred: str, ref: str) -> float:
    """Compute token-level overlap for location strings.

    Uses Counter-based (multiset) intersection so that repeating a single
    matching token dilutes precision, consistent with all other token
    overlap functions in the codebase.

    Args:
        pred: Predicted location
        ref: Reference location

    Returns:
        Token F1 score
    """
    from collections import Counter

    # Remove common non-informative tokens
    stopwords = {"region", "area", "zone", "side", "aspect"}

    pred_counts = Counter(t for t in normalize_location(pred).split() if t not in stopwords)
    ref_counts = Counter(t for t in normalize_location(ref).split() if t not in stopwords)

    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())

    if not pred_total or not ref_total:
        return 1.0 if pred_total == ref_total else 0.0

    intersection = sum((pred_counts & ref_counts).values())

    if not intersection:
        return 0.0

    precision = intersection / pred_total
    recall = intersection / ref_total

    return 2 * precision * recall / (precision + recall)


@beartype
def compute_location_reward(
    prediction: str,
    reference: str,
) -> dict[str, float | str | None]:
    """Compute location reference verification reward.

    Combines:
    - Exact region match (canonical names)
    - Hierarchical match (parent/child relationships)
    - Token overlap

    Args:
        prediction: Predicted location reference
        reference: Ground truth location reference

    Returns:
        Dict with component scores and final reward
    """
    pred_norm = normalize_location(prediction)
    ref_norm = normalize_location(reference)

    # Compute component scores
    exact = compute_region_match(pred_norm, ref_norm)
    hierarchy = compute_hierarchy_match(pred_norm, ref_norm)
    token = compute_token_match(pred_norm, ref_norm)

    # Weighted combination
    weights = {"exact": 0.5, "hierarchy": 0.3, "token": 0.2}

    reward = weights["exact"] * exact + weights["hierarchy"] * hierarchy + weights["token"] * token

    return {
        "exact_match": exact,
        "hierarchy_match": hierarchy,
        "token_overlap": token,
        "reward": reward,
        "pred_canonical": get_canonical_region(pred_norm),
        "ref_canonical": get_canonical_region(ref_norm),
    }


@beartype
def compute_batch_location_rewards(
    predictions: Sequence[str],
    references: Sequence[str],
) -> list[dict[str, float | str | None]]:
    """Compute location rewards for a batch of samples.

    Args:
        predictions: List of predicted locations
        references: List of reference locations

    Returns:
        List of reward dicts for each sample
    """
    return [
        compute_location_reward(pred, ref)
        for pred, ref in zip(predictions, references, strict=True)
    ]
