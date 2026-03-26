import asyncio
from collections.abc import Callable

import numpy as np

from k_steering.evals.judges.ood import OODJudge


async def is_ood(
    texts: list[str],
    *,
    judge: OODJudge,
    frac: float = 5.0,
    score_thresh: float = 50.0,
    verbose: bool = False,
) -> bool:
    """
    Helper function for checking if the steered texts are coherent.

    Args:
        texts (List[str]): List of Text to be evaluated.
        judge (OODJudge): Coherence Judge
        frac (float, optional): Proportion threshold for having bad scores. Defaults to 5.0.
        score_thresh (float, optional): Score threshold for bad qualification. Defaults to 50.0.
        verbose (bool, optional): Verbsoity Boolean. Defaults to False.

    Returns:
        bool: _description_
    """
    print(f"Evaluating Generated responses using {judge.judge_name}")
    scores = await judge.evaluate_batch(generations=texts)
    scores = np.array([0.0 if s is None else float(s) for s in scores])
    bad = scores < score_thresh
    frac_bad = 100.0 * bad.mean()
    if verbose:
        print(
            f"Judge mean={scores.mean():.1f} | {bad.sum()}/{len(texts)} below {score_thresh} (" f"{frac_bad:.1f}% → {'OOD' if frac_bad > frac else 'OK'})"
        )
    return frac_bad > frac


async def calibrate_alpha_ood_only(
    ood_check_async: Callable[[float], 'asyncio.Future'],
    *,
    min_alpha: float = 1.0,
    max_alpha: float = 32.0,
    tol: float = 0.05,
    max_iters: int = 8,
) -> float:
    """
    Calibrating the Optimum Steering Strength using binary search.

    Args:
        ood_check_async (Callable[[float], asyncio.Future]): Helper function for coherence check.
        min_alpha (float, optional): Minimum Allowable Steering Strength. Defaults to 1.0.
        max_alpha (float, optional): Maximum Alowable Steering Strength. Defaults to 32.0.
        tol (float, optional): Tolerance Limit. Defaults to 0.05.
        max_iters (int, optional): Maximum Allowable Iterations. Defaults to 8.

    Returns:
        float: _description_
    """
    lo, hi = min_alpha, max_alpha
    last_good = min_alpha
    for _ in range(max_iters):
        print(f"Parameter Sweep Iteration No: {_}")
        if hi / lo <= 1 + tol:
            break
        mid = (lo + hi) / 2
        if await ood_check_async(mid):
            hi = mid
        else:
            last_good = mid
            lo = mid
    return float(last_good)

