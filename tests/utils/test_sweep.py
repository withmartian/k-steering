import pytest
import numpy as np

from k_steering.utils.sweep import is_ood, calibrate_alpha_ood_only


class DummyJudge:
    def __init__(self, scores):
        self.scores = scores
        self.judge_name = "dummy-judge"

    async def evaluate_batch(self, generations):
        assert len(generations) == len(self.scores)
        return self.scores


@pytest.mark.asyncio
async def test_is_ood_all_good():
    judge = DummyJudge(scores=[80, 90, 75, 100])
    texts = ["a", "b", "c", "d"]

    result = await is_ood(
        texts,
        judge=judge,
        frac=5.0,
        score_thresh=50.0,
    )
    assert result == False


@pytest.mark.asyncio
async def test_is_ood_all_bad():
    judge = DummyJudge(scores=[10, 20, 30, 40])
    texts = ["a", "b", "c", "d"]

    result = await is_ood(
        texts,
        judge=judge,
        frac=5.0,          # >5% bad triggers OOD
        score_thresh=50.0,
    )

    assert result == True


@pytest.mark.asyncio
async def test_is_ood_mixed_fraction_boundary():
    # 1 bad out of 4 → 25%
    judge = DummyJudge(scores=[80, 90, 20, 100])
    texts = ["a", "b", "c", "d"]

    result = await is_ood(
        texts,
        judge=judge,
        frac=30.0,
        score_thresh=50.0,
    )

    assert result == False


@pytest.mark.asyncio
async def test_is_ood_handles_none_scores():
    judge = DummyJudge(scores=[None, 80, None, 90])
    texts = ["a", "b", "c", "d"]

    # None → 0.0 → below threshold
    result = await is_ood(
        texts,
        judge=judge,
        frac=40.0,
        score_thresh=50.0,
    )

    # 2 / 4 bad = 50%
    assert result == True


@pytest.mark.asyncio
async def test_calibrate_alpha_finds_boundary():
    """
    OOD happens for alpha >= 8.0
    Expect last_good < 8.0
    """

    async def mock_ood(alpha: float) -> bool:
        return alpha >= 8.0

    alpha = await calibrate_alpha_ood_only(
        mock_ood,
        min_alpha=1.0,
        max_alpha=32.0,
        tol=0.05,
        max_iters=10,
    )

    assert alpha < 8.0
    assert alpha >= 1.0


@pytest.mark.asyncio
async def test_calibrate_alpha_no_ood_anywhere():
    async def mock_ood(alpha: float) -> bool:
        return False  # never OOD

    alpha = await calibrate_alpha_ood_only(
        mock_ood,
        min_alpha=1.0,
        max_alpha=16.0,
    )

    # Should push upward
    assert alpha > 8.0


@pytest.mark.asyncio
async def test_calibrate_alpha_immediate_ood():
    async def mock_ood(alpha: float) -> bool:
        return True  # always OOD

    alpha = await calibrate_alpha_ood_only(
        mock_ood,
        min_alpha=2.0,
        max_alpha=16.0,
    )

    # last_good initialized to min_alpha
    assert alpha == 2.0


@pytest.mark.asyncio
async def test_calibrate_alpha_respects_max_iters():
    calls = []

    async def mock_ood(alpha: float) -> bool:
        calls.append(alpha)
        return alpha > 10.0

    await calibrate_alpha_ood_only(
        mock_ood,
        max_iters=3,
    )

    assert len(calls) <= 3
