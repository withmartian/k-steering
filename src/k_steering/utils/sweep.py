from typing import List, Callable
import asyncio
import numpy as np

from src.k_steering.evals.judges.base_judge import BaseLLMJudge


async def is_ood(
    texts: List[str],
    *,
    judge: BaseLLMJudge,
    frac: float = 5.0,
    score_thresh: float = 50.0,
    verbose: bool = False,
) -> bool:
    scores = await asyncio.gather(*[judge.judge(generation=t) for t in texts])
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
    lo, hi = min_alpha, max_alpha
    last_good = min_alpha
    for _ in range(max_iters):
        if hi / lo <= 1 + tol:
            break
        mid = (lo + hi) / 2
        if await ood_check_async(mid):
            hi = mid
        else:
            last_good = mid
            lo = mid
    return float(last_good)

