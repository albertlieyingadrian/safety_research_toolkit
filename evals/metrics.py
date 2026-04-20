"""
Evaluation metrics and confidence intervals.
"""

import re
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap
from statsmodels.stats.proportion import proportion_confint


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_ci(
    n_successes: int,
    n_total: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Args:
        n_successes: Number of positive outcomes.
        n_total: Total observations.
        alpha: Significance level (0.05 = 95% CI).

    Returns:
        ``(lower, upper)`` as fractions in ``[0, 1]``.
    """
    lo, hi = proportion_confint(n_successes, n_total, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Optional[Callable] = None,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> Tuple[float, float]:
    """Bootstrap percentile CI for a statistic of *data*.

    Args:
        data: 1-D array of observations.
        stat_fn: Statistic to bootstrap (default: ``np.mean``).
        n_resamples: Bootstrap iterations.
        confidence_level: E.g. 0.95 for a 95% CI.
        random_state: RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` in the same units as *stat_fn* output.
    """
    if stat_fn is None:
        stat_fn = np.mean
    result = scipy_bootstrap(
        (data,),
        stat_fn,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        random_state=random_state,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def extract_answer(judge_text: str, answer_tag: str = "answer") -> Optional[str]:
    """Extract the text inside ``<answer_tag>…</answer_tag>`` tags.

    Returns the lowercased content, or ``None`` if not found.
    """
    match = re.search(rf"<{answer_tag}>(.*?)</{answer_tag}>", judge_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip().lower() if match else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _bootstrap_sample_sizes(n_total: int) -> List[int]:
    sizes = []
    s = 100
    while s < n_total:
        sizes.append(s)
        s += 100
    sizes.append(n_total)
    return sizes


def _bootstrap_for_size(binary_outcomes: np.ndarray, size: int) -> dict:
    data = binary_outcomes[:size]
    lo, hi = bootstrap_ci(data, n_resamples=100_000)
    rate = np.mean(data) * 100
    return {
        "sample_size": size,
        "rate": f"{rate:.2f}%",
        "ci_95": {
            "lower": f"{lo * 100:.2f}",
            "upper": f"{hi * 100:.2f}",
        },
    }


def compute_judge_metrics(
    judge_responses: List,
    answer_tag: str = "answer",
    answer_match: str = "yes",
) -> dict:
    """Compute positivity rate + Wilson & bootstrap CIs from judge output pairs.

    Args:
        judge_responses: List of judge response strings.
        answer_tag: XML tag to extract the answer from.
        answer_match: Value counted as a positive.

    Returns:
        Dict with keys ``total``, ``present``, ``positives``, ``ci_95``,
        ``bootstrap``.
    """
    answers = [extract_answer(response, answer_tag) for response in judge_responses]
    present = sum(a is not None for a in answers)
    n_yes = answers.count(answer_match)
    rate = n_yes / present if present else 0.0

    ci_lo, ci_hi = wilson_ci(n_yes, present)

    binary = np.array([1 if a == answer_match else 0 for a in answers if a is not None])
    sample_sizes = _bootstrap_sample_sizes(len(binary))
    bootstrap_results = [_bootstrap_for_size(binary, s) for s in sample_sizes]

    return {
        "total": len(answers),
        "present": present,
        "positives": f"{rate * 100:.2f}%",
        "ci_95": {
            "lower": f"{ci_lo * 100:.2f}",
            "upper": f"{ci_hi * 100:.2f}",
        },
        "bootstrap": bootstrap_results,
    }
