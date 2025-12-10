"""Tests for verifiers integration utilities."""

from __future__ import annotations

import json

import verifiers as vf

# Example environments
from examples.gemex_thinkvg.src.verifiers.environment import GEMeXThinkVGToolEnv
from examples.gemex_thinkvg.src.verifiers.environment import load_environment as load_gemex_env
from examples.verifiers_integration.nova_example import NOVAToolEnv
from radiant_harness.verifiers.rewards import CombinedReward
from radiant_harness.verifiers.rewards import ExactMatchReward
from radiant_harness.verifiers.rewards import IoUReward
from radiant_harness.verifiers.rewards import TokenF1Reward


class TestExactMatchReward:
    """Test exact match reward function."""

    def test_basic_match(self) -> None:
        reward = ExactMatchReward(normalize=False, case_sensitive=True)

        # Exact match
        assert reward("", "Hello", {"answer": "Hello"}) == 1.0

        # No match
        assert reward("", "Hello", {"answer": "World"}) == 0.0

    def test_normalized_match(self) -> None:
        reward = ExactMatchReward(normalize=True, case_sensitive=False)

        # Different case
        assert reward("", "hello", {"answer": "Hello"}) == 1.0

        # With braces
        assert reward("", "{Hello}", {"answer": "Hello"}) == 1.0

        # With whitespace
        assert reward("", "  Hello  ", {"answer": "Hello"}) == 1.0

    def test_from_list_completion(self) -> None:
        reward = ExactMatchReward()

        completion = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Answer: Paris"},
        ]

        assert reward("", completion, {"answer": "Paris"}) == 0.0  # "Answer: " != "Paris"


class TestTokenF1Reward:
    """Test token F1 reward function."""

    def test_perfect_match(self) -> None:
        reward = TokenF1Reward()

        score = reward("", "The quick brown fox", {"answer": "The quick brown fox"})
        assert score == 1.0

    def test_partial_match(self) -> None:
        reward = TokenF1Reward()

        score = reward("", "The brown fox", {"answer": "The quick brown fox"})
        # Tokens: ["the", "brown", "fox"] vs ["the", "quick", "brown", "fox"]
        # Intersection: 3, Union: 4, F1 = 2*3/(3+4) = 6/7 ≈ 0.857
        assert abs(score - 0.857) < 0.01

    def test_no_match(self) -> None:
        reward = TokenF1Reward()

        score = reward("", "Hello world", {"answer": "Goodbye world"})
        # Tokens with normalize=True: ["hello", "world"] vs ["goodbye", "world"]
        # Intersection: 1, Precision: 1/2=0.5, Recall: 1/2=0.5
        # F1 = 2*0.5*0.5/(0.5+0.5) = 0.5
        assert abs(score - 0.5) < 1e-6

    def test_empty_strings(self) -> None:
        reward = TokenF1Reward()

        # Both empty
        assert reward("", "", {"answer": ""}) == 1.0

        # One empty
        assert reward("", "Hello", {"answer": ""}) == 0.0


class TestIoUReward:
    """Test IoU reward function."""

    def test_perfect_overlap(self) -> None:
        reward = IoUReward()

        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]

        # Test with bbox in completion and reference
        completion = '{"bbox": [0, 0, 10, 10]}'
        assert reward("", completion, {"bbox": box1}) == 1.0

        # Test with same boxes
        completion = '{"bbox": [0, 0, 10, 10]}'
        assert reward("", completion, {"bbox": box2}) == 1.0

    def test_no_overlap(self) -> None:
        reward = IoUReward()

        box1 = [0, 0, 10, 10]

        score = reward("", None, {"bbox": box1})
        # Should extract from completion if no bbox in info
        assert score == 0.0  # No bbox found

    def test_partial_overlap(self) -> None:
        reward = IoUReward()

        # 1/3 overlap (area 25 out of union 75)
        box1 = [0, 0, 10, 10]  # Area = 100
        box2 = [5, 0, 15, 10]  # Area = 100
        # Intersection = 5*10 = 50
        # Union = 100 + 100 - 50 = 150
        # IoU = 50/150 = 1/3 ≈ 0.333

        # Test with JSON completion
        completion = {"bbox": box2}
        score = reward("", json.dumps(completion), {"bbox": box1})
        assert abs(score - 0.333) < 0.01


class TestCombinedReward:
    """Test combined reward function."""

    def test_single_reward(self) -> None:
        exact = ExactMatchReward()
        combined = CombinedReward(rewards=[exact])

        score = combined("", "Hello", {"answer": "Hello"})
        assert score == 1.0

    def test_multiple_rewards(self) -> None:
        exact = ExactMatchReward()
        f1 = TokenF1Reward()
        combined = CombinedReward(
            rewards=[exact, f1],
            weights=[0.6, 0.4],
        )

        # Test where exact=0, f1=0.5
        score = combined("", "Hello world", {"answer": "Hello"})

        # Manually compute expected
        exact_score = exact("", "Hello world", {"answer": "Hello"})  # 0.0
        f1_score = f1("", "Hello world", {"answer": "Hello"})  # 0.5
        expected = 0.6 * exact_score + 0.4 * f1_score  # 0.2

        score = combined("", "Hello world", {"answer": "Hello"})
        assert abs(score - expected) < 1e-6

    def test_weight_normalization(self) -> None:
        exact = ExactMatchReward()
        f1 = TokenF1Reward()

        # Weights don't sum to 1, should normalize
        combined = CombinedReward(
            rewards=[exact, f1],
            weights=[3, 2],  # Sum to 5, should normalize to 0.6, 0.4
        )

        assert combined.weights == [0.6, 0.4]


def test_verifiers_optional() -> None:
    """Test that verifiers integration handles missing import gracefully."""
    # This should not raise an error even if verifiers is not installed
    from radiant_harness.verifiers import BaseRewardFunction

    assert BaseRewardFunction is not None


class TestToolEnvIntegrations:
    """Smoke tests for ToolEnv-based example environments."""

    def test_gemex_toolenv_setup(self) -> None:
        cases = [
            {
                "question": "Is there a pleural effusion?",
                "question_type": "open_ended",
                "image_url": "https://example.com/xray.jpg",
                "answer": "No effusion",
                "location_reference": "right lower lobe",
                "bbox": [10, 20, 30, 40],
            }
        ]

        env = load_gemex_env(cases=cases, max_turns=3)

        assert isinstance(env, GEMeXThinkVGToolEnv)
        assert len(env.dataset) == 1
        assert hasattr(env, "tools")
        assert len(env.tools) == 5  # zoom, crop, contrast, threshold, search
        assert isinstance(env.rubric, vf.Rubric)
        state = env.build_initial_state(env.dataset[0]["prompt"], env.dataset[0]["info"])
        assert state["turn"] == 0

    def test_nova_toolenv_setup(self) -> None:
        cases = [
            {
                "question": "Identify abnormalities",
                "findings": "Small infarct",
                "location": "left basal ganglia",
                "bbox": [1, 2, 3, 4],
                "severity": "mild",
                "image_path": "https://example.com/mri.jpg",
            }
        ]

        env = NOVAToolEnv(cases=cases, max_turns=2, enable_tools=True)

        assert isinstance(env, vf.ToolEnv)
        assert len(env.dataset) == 1
        assert hasattr(env, "tools")
        assert len(env.tools) == 4  # zoom, crop, contrast, search
        assert env._max_turns == 2
