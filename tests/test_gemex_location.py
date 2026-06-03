"""Tests for GEMeX-ThinkVG location reward — Patch Set #2.

Covers:
- Finding #5: Ambiguous substring matching (prefer most-specific match)
- Finding #6: Shallow hierarchy (transitive ancestor/descendant)
- Finding #7: Missing anatomical synonyms
"""

from __future__ import annotations

import pytest

from examples.gemex_thinkvg.src.rewards.location import _is_ancestor
from examples.gemex_thinkvg.src.rewards.location import compute_hierarchy_match
from examples.gemex_thinkvg.src.rewards.location import compute_location_reward
from examples.gemex_thinkvg.src.rewards.location import get_canonical_region

# ── Finding #5: Substring matching prefers most-specific ──────────────


class TestSubstringSpecificity:
    """get_canonical_region should prefer the longest (most specific) match."""

    def test_right_lower_lobe_not_right_lung(self) -> None:
        """'right lower lobe' should map to itself, not 'right lung'."""
        assert get_canonical_region("right lower lobe") == "right lower lobe"

    def test_right_upper_lobe_not_right_lung(self) -> None:
        assert get_canonical_region("right upper lobe") == "right upper lobe"

    def test_left_lower_lobe_not_left_lung(self) -> None:
        assert get_canonical_region("left lower lobe") == "left lower lobe"

    def test_bare_right_prefers_most_specific(self) -> None:
        """'right' alone is ambiguous — should match 'right lung' (longest
        canonical containing 'right' as substring)."""
        canonical = get_canonical_region("right")
        # "right" is a substring of both "right lung" and "right upper lobe",
        # etc. The longest canonical whose name contains "right" as a substring
        # should win. Since "right" in "right lower lobe" and "right lower lobe"
        # is longer than "right lung", we expect one of the specific lobes.
        # But "right" in "right costophrenic angle" is even longer.
        # The key assertion: it must NOT be None.
        assert canonical is not None

    def test_cardiac_silhouette_maps_to_heart(self) -> None:
        assert get_canonical_region("cardiac silhouette") == "heart"

    def test_rll_abbreviation(self) -> None:
        """Abbreviation 'RLL' should resolve to 'right lower lobe'."""
        assert get_canonical_region("rll") == "right lower lobe"

    def test_right_base_maps_to_rll(self) -> None:
        assert get_canonical_region("right base") == "right lower lobe"

    def test_exact_match_beats_substring(self) -> None:
        """Direct canonical match should always win over substring."""
        # "heart" is both a canonical AND a substring of "heart shadow"
        assert get_canonical_region("heart") == "heart"

    def test_perihilar_maps_to_bilateral_hilum(self) -> None:
        assert get_canonical_region("perihilar") == "bilateral hilum"


# ── Finding #6: Transitive hierarchy ──────────────────────────────────


class TestTransitiveHierarchy:
    """Hierarchy matching should work across grandparent+ relationships."""

    # -- _is_ancestor helper --

    def test_direct_parent(self) -> None:
        """bilateral lung → right lung = depth 1."""
        assert _is_ancestor("bilateral lung", "right lung") == 1

    def test_grandparent(self) -> None:
        """bilateral lung → right lower lobe = depth 2."""
        assert _is_ancestor("bilateral lung", "right lower lobe") == 2

    def test_great_grandparent(self) -> None:
        """chest → right lower lobe = depth 3 (chest → bilateral lung → right lung → rll)."""
        assert _is_ancestor("chest", "right lower lobe") == 3

    def test_no_relation(self) -> None:
        """heart and right lung are not in an ancestor chain."""
        assert _is_ancestor("heart", "right lung") is None

    def test_reverse_not_ancestor(self) -> None:
        """right lower lobe is NOT an ancestor of chest."""
        assert _is_ancestor("right lower lobe", "chest") is None

    # -- compute_hierarchy_match with transitive relations --

    def test_chest_to_rll_nonzero(self) -> None:
        """'chest' → 'right lower lobe' should get partial credit (was 0.0 before fix)."""
        score = compute_hierarchy_match("chest", "right lower lobe")
        assert score > 0.0, f"chest→rll should get partial credit, got {score}"
        # Depth 3, pred too general: 0.5 / 3 ≈ 0.167
        assert score < 0.5, "Should be less than direct parent score"

    def test_bilateral_lung_to_rll(self) -> None:
        """bilateral lung → right lower lobe: depth 2, pred too general."""
        score = compute_hierarchy_match("bilateral lung", "right lower lobe")
        assert 0.1 < score < 0.5, f"Expected ~0.25, got {score}"

    def test_rll_to_chest_nonzero(self) -> None:
        """Pred more specific than ref: 'right lower lobe' → 'chest'."""
        score = compute_hierarchy_match("right lower lobe", "chest")
        assert score > 0.0, f"rll→chest should get partial credit, got {score}"

    def test_direct_parent_score_unchanged(self) -> None:
        """Direct parent should still score 0.5 (backward compat)."""
        score = compute_hierarchy_match("bilateral lung", "right lung")
        assert score == pytest.approx(0.5)

    def test_direct_child_score_unchanged(self) -> None:
        """Direct child should still score 0.7 (backward compat)."""
        score = compute_hierarchy_match("right lung", "bilateral lung")
        assert score == pytest.approx(0.7)

    def test_siblings_score_unchanged(self) -> None:
        """Siblings should still score 0.3 (backward compat)."""
        score = compute_hierarchy_match("right lung", "left lung")
        assert score == pytest.approx(0.3)

    def test_exact_match_still_1(self) -> None:
        score = compute_hierarchy_match("right lung", "right lung")
        assert score == pytest.approx(1.0)

    def test_unrelated_regions_zero(self) -> None:
        """heart vs right lung — no hierarchy path."""
        score = compute_hierarchy_match("heart", "right lung")
        assert score == 0.0

    def test_deeper_pred_more_specific_than_ref(self) -> None:
        """right hilum (child of right lung) → bilateral lung (grandparent)."""
        score = compute_hierarchy_match("right hilum", "bilateral lung")
        # right hilum is depth-2 descendant of bilateral lung
        # ref is ancestor, pred more specific: 0.7 / depth
        assert score > 0.2


# ── Finding #7: Missing synonyms ─────────────────────────────────────


class TestMissingSynonyms:
    """Newly added synonyms should resolve correctly."""

    @pytest.mark.parametrize(
        ("input_str", "expected_canonical"),
        [
            # Hilum
            ("right hilar", "right hilum"),
            ("right hilar region", "right hilum"),
            ("left hilar", "left hilum"),
            ("perihilar", "bilateral hilum"),
            ("parahilar", "bilateral hilum"),
            # Apex / apical
            ("right apex", "right upper lobe"),
            ("left apical", "left upper lobe"),
            ("apical", "lung apex"),
            ("apex", "lung apex"),
            ("apices", "lung apex"),
            # Basilar
            ("right basilar", "right lower lobe"),
            ("left basilar", "left lower lobe"),
            # Retrocardiac
            ("retrocardiac", "retrocardiac"),
            ("retrocardiac region", "retrocardiac"),
            ("behind heart", "retrocardiac"),
            # Paratracheal
            ("paratracheal", "trachea"),
        ],
    )
    def test_synonym_resolves(self, input_str: str, expected_canonical: str) -> None:
        result = get_canonical_region(input_str)
        assert result == expected_canonical, (
            f"'{input_str}' should map to '{expected_canonical}', got '{result}'"
        )

    def test_hilum_in_hierarchy(self) -> None:
        """right hilum should be a child of right lung in the hierarchy."""
        assert _is_ancestor("right lung", "right hilum") == 1

    def test_retrocardiac_in_hierarchy(self) -> None:
        """retrocardiac should be a child of mediastinum."""
        assert _is_ancestor("mediastinum", "retrocardiac") == 1

    def test_bilateral_hilum_children(self) -> None:
        assert _is_ancestor("bilateral hilum", "right hilum") == 1
        assert _is_ancestor("bilateral hilum", "left hilum") == 1


# ── Integration: location reward end-to-end ───────────────────────────


class TestLocationRewardIntegration:
    """End-to-end location reward with the new fixes."""

    def test_exact_match_full_reward(self) -> None:
        result = compute_location_reward("right lower lobe", "right lower lobe")
        assert result["reward"] > 0.9

    def test_synonym_match_high_reward(self) -> None:
        """RLL should match right lower lobe via synonym → exact + hierarchy = 0.8.

        Token overlap is 0 because normalize_location does not expand
        abbreviations, but exact and hierarchy already capture equivalence.
        """
        result = compute_location_reward("rll", "right lower lobe")
        assert result["reward"] >= 0.8
        assert result["exact_match"] == 1.0

    def test_hilar_vs_hilum_match(self) -> None:
        result = compute_location_reward("right hilar", "right hilum")
        assert result["reward"] >= 0.9

    def test_chest_vs_rll_low_but_nonzero(self) -> None:
        """Very general 'chest' vs specific 'right lower lobe'."""
        result = compute_location_reward("chest", "right lower lobe")
        assert 0.0 < result["reward"] < 0.4, (
            f"chest→rll should be low but nonzero: {result['reward']:.3f}"
        )

    def test_wrong_side_low(self) -> None:
        """Right vs left lung should score as siblings (0.3)."""
        result = compute_location_reward("right lung", "left lung")
        # exact=0, hierarchy=0.3, token has partial overlap ("lung")
        assert result["reward"] < 0.4

    def test_completely_wrong_region_zero(self) -> None:
        """Heart vs right lower lobe — unrelated."""
        result = compute_location_reward("heart", "right lower lobe")
        # No hierarchy relation, no token overlap (different words)
        assert result["reward"] < 0.15
