"""Recompute Section 7 of the revised quantum-hybrid-attack manuscript.

This implementation keeps the manuscript's original optimization objective:
the sum of the classical BKZ proxy and QSearch Toffoli depth, divided by the
Babai success probability.  Once that optimum is selected, Toffoli count,
total logical gate depth, and total logical gate count are evaluated at the
same point.

Corrections relative to the earlier notebook
---------------------------------------------
* the Babai residual uses the original CBD second moment, ``eta/2``;
* the reduced basis dimension is ``k=2*n-r``;
* the modified-GSA exponents are ``t-2*j+1`` and sum to zero;
* ``log2(L_j)`` is passed to the circuit code, not ``log2(sqrt(L_j))``;
* the constant-division convention and bit lengths come from the revised
  :mod:`Q_Toffoli_cost` module;
* the success probability is evaluated with a regularized incomplete beta
  function instead of one numerical quadrature per basis vector.

The exhaustive block-size search uses every integer ``beta`` from 91 through
``k`` that satisfies the modified-GSA conditions.  Parallelism changes only
runtime, not the set of evaluated candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import betainc

import Q_Toffoli_cost as QTC
import Q_gate_cost as QGC
import FT_gate_cost as QFT


MODEL_VERSION = "2026-08-15-k-dimension-gsa-det-cbd-moment-v1"
DEFAULT_Q = 3329


def cbd_second_moment(eta: int) -> float:
    r"""Return ``E[X^2]=eta/2`` for ``X ~ CBD_eta``.

    Only the guessed register is prepared according to the transformed
    distribution ``T(CBD_eta)``.  The Babai residual contains the unguessed
    secret entries and error entries, both of which follow the original CBD.
    """

    if eta not in (2, 3):
        raise ValueError("eta must be 2 or 3")
    return eta / 2.0


def calculate_delta(block_size: int) -> float:
    """Root-Hermite factor used by the manuscript's BKZ model."""

    if block_size <= 2:
        raise ValueError("block_size must exceed 2")
    return (
        ((math.pi * block_size) ** (1.0 / block_size) * block_size)
        / (2.0 * math.pi * math.e)
    ) ** (1.0 / (2.0 * (block_size - 1)))


def calculate_reduction_runtime_log2(
    block_size: int,
    reduced_dimension: int,
    *,
    sieve_exponent: float = 0.292,
) -> float:
    """Return the base-2 logarithm of the classical BDGL/BKZ proxy."""

    if block_size <= 90:
        raise ValueError("the stated BDGL fit requires block_size > 90")
    if block_size > reduced_dimension:
        raise ValueError("block_size cannot exceed the reduced dimension")
    rho = math.ceil(
        (reduced_dimension**2 / block_size**2)
        * math.log2(reduced_dimension)
    )
    return (
        math.log2(rho)
        + math.log2(reduced_dimension)
        + sieve_exponent * block_size
        + 16.4
    )


def modified_gsa_profile_log2(
    reduced_dimension: int,
    q_rank: int,
    delta: float,
    *,
    q: int = DEFAULT_Q,
    verify_determinant: bool = False,
) -> tuple[np.ndarray, int]:
    r"""Return ``log2(||b_tilde_i||)`` and the GSA-tail dimension.

    The first ``reduced_dimension-tail_dimension`` lengths equal ``q``.  In
    the tail, the delta exponents are

    ``tail_dimension - 2*j + 1``, ``j=1,...,tail_dimension``.

    They sum to zero.  Together with the q-scaling term this makes the product
    of all Gram--Schmidt lengths exactly ``q**(reduced_dimension-q_rank)``;
    for the Kyber embedding this exponent is ``n``.
    """

    if not 0 < q_rank < reduced_dimension:
        raise ValueError("require 0 < q_rank < reduced_dimension")
    if delta <= 1.0:
        raise ValueError("delta must exceed one")

    log_delta_base_q = math.log(delta) / math.log(q)
    tail_dimension = min(
        math.floor(math.sqrt(q_rank / log_delta_base_q)),
        reduced_dimension,
    )
    if not q_rank < tail_dimension < reduced_dimension:
        raise ValueError("modified-GSA tail condition is not satisfied")

    prefix = reduced_dimension - tail_dimension
    log2_q = math.log2(q)
    j = np.arange(1, tail_dimension + 1, dtype=np.float64)
    delta_exponents = tail_dimension - 2.0 * j + 1.0
    tail = (
        delta_exponents * math.log2(delta)
        + ((tail_dimension - q_rank) / tail_dimension) * log2_q
    )
    profile = np.concatenate(
        (np.full(prefix, log2_q, dtype=np.float64), tail)
    )

    if verify_determinant:
        expected = (reduced_dimension - q_rank) * log2_q
        if not math.isclose(
            float(profile.sum()), expected, rel_tol=0.0, abs_tol=1e-8
        ):
            raise AssertionError("modified-GSA profile violates determinant")
    return profile, tail_dimension


def babai_success_probability_log2(
    log2_gs_lengths: Sequence[float],
    residual_second_moment: float,
) -> float:
    r"""Return the logarithm of the Babai success-probability heuristic.

    For reduced dimension ``k`` and
    ``x_i=min{1,(||b_tilde_i||/(2Y))^2}``, each factor equals

    ``I_{x_i}(1/2,(k-1)/2)``,

    where ``I`` is the regularized incomplete beta function and
    ``Y=sqrt(residual_second_moment*k)``.
    """

    logs = np.asarray(log2_gs_lengths, dtype=np.float64)
    k = int(logs.size)
    if k < 2:
        raise ValueError("the reduced dimension must be at least two")
    if residual_second_moment <= 0:
        raise ValueError("residual_second_moment must be positive")

    y_norm = math.sqrt(residual_second_moment * k)
    ratios = np.exp2(logs) / (2.0 * y_norm)
    x = np.minimum(ratios * ratios, 1.0)
    factors = betainc(0.5, (k - 1.0) / 2.0, x)
    if np.any(factors <= 0.0):
        return float("-inf")
    return float(np.log2(factors).sum())


def _toffoli_depth_from_profile_log2(
    n: int, r: int, log2_gs_lengths: np.ndarray
) -> float:
    """Return ``log2(L*TD(Q))`` using the exact grouped depth formula."""

    log2_L_values = 2.0 * log2_gs_lengths
    floors = np.floor(log2_L_values).astype(np.int64)
    unique, counts = np.unique(floors, return_counts=True)
    multiplicities = {
        int(value): int(count) for value, count in zip(unique, counts)
    }
    q_depth = QTC.compute_q_toffoli_depth_grouped(
        n,
        r,
        first_floor=int(floors[0]),
        floor_multiplicities=multiplicities,
    )
    log2_repetitions = QTC.kyber_instance_parameters(n)[0] * r
    return log2_repetitions + math.log2(q_depth)


def log2_sum_stable(x: float, y: float) -> float:
    """Return ``log2(2**x+2**y)`` without overflow."""

    high, low = max(x, y), min(x, y)
    return high + math.log2(1.0 + math.exp2(low - high))


def evaluate_candidate(
    n: int,
    r: int,
    block_size: int,
    *,
    q: int = DEFAULT_Q,
) -> dict | None:
    """Evaluate one ``(n,r,block_size)`` point, or return ``None``."""

    k = QTC.compute_k(n, r)
    q_rank = n - r
    if q_rank <= 0 or not 91 <= block_size <= k:
        return None

    delta = calculate_delta(block_size)
    try:
        log2_gs, tail_dimension = modified_gsa_profile_log2(
            k, q_rank, delta, q=q
        )
    except ValueError:
        return None

    eta_1 = QTC.kyber_instance_parameters(n)[1]
    prob_log2 = babai_success_probability_log2(
        log2_gs, cbd_second_moment(eta_1)
    )
    if not math.isfinite(prob_log2):
        return None

    reduction_log2 = calculate_reduction_runtime_log2(block_size, k)
    hybrid_depth_log2 = _toffoli_depth_from_profile_log2(n, r, log2_gs)
    total_log2 = log2_sum_stable(
        reduction_log2 - prob_log2,
        hybrid_depth_log2 - prob_log2,
    )
    return {
        "model_version": MODEL_VERSION,
        "n": n,
        "r": r,
        "reduced_dimension": k,
        "block_size": block_size,
        "gsa_tail_dimension": tail_dimension,
        "delta": delta,
        "prob_log2": prob_log2,
        "T_red_log2": reduction_log2,
        "T_hyb_log2": hybrid_depth_log2,
        "total_cost_log2": total_log2,
    }


def search_best_objective_for_r(
    n: int,
    r: int,
    *,
    q: int = DEFAULT_Q,
    block_sizes: Iterable[int] | None = None,
) -> dict:
    """Exhaustively minimize the manuscript objective for one fixed ``r``."""

    k = QTC.compute_k(n, r)
    if block_sizes is None:
        block_sizes = range(91, k + 1)

    best: dict | None = None
    evaluated = 0
    for block_size in block_sizes:
        candidate = evaluate_candidate(n, r, int(block_size), q=q)
        if candidate is None:
            continue
        evaluated += 1
        if best is None or candidate["total_cost_log2"] < best["total_cost_log2"]:
            best = candidate

    if best is None:
        return {
            "model_version": MODEL_VERSION,
            "n": n,
            "r": r,
            "valid": False,
            "evaluated_candidates": 0,
        }
    return {**best, "valid": True, "evaluated_candidates": evaluated}


def _search_task(args: tuple[int, int, int]) -> dict:
    n, r, q = args
    return search_best_objective_for_r(n, r, q=q)


def _profile_at_candidate(candidate: dict, q: int) -> np.ndarray:
    n, r = int(candidate["n"]), int(candidate["r"])
    k = QTC.compute_k(n, r)
    q_rank = n - r
    delta = float(candidate["delta"])
    profile, tail = modified_gsa_profile_log2(
        k, q_rank, delta, q=q, verify_determinant=True
    )
    if tail != int(candidate["gsa_tail_dimension"]):
        raise AssertionError("stored GSA-tail dimension is inconsistent")
    return profile


def total_qubit_upper_bound(n: int, r: int) -> int:
    """Evaluate the corrected appendix qubit bound directly from lengths."""

    f = QTC.kyber_instance_parameters(n)[0]
    k = QTC.compute_k(n, r)
    l_r = QTC.compute_l_r(r)
    l_k = QTC.compute_l_k(k)
    l_t = QTC.compute_l_t(l_r)
    l_tr = QTC.compute_l_tr(l_r)
    l_p = QTC.compute_l_p(l_t, f, r, k)
    l_tilde_b = QTC.compute_l_tilde_b(l_p)
    l_t_tilde_b = QTC.compute_l_t_tilde_b(l_t, l_tilde_b)
    l_M = QTC.compute_l_M(l_t_tilde_b, l_k)
    max_l_u = l_M - QTC.compute_l_L_j(l_p, 0.0)
    return (
        2 * k * l_t_tilde_b
        + k * l_tr
        + k * l_t
        + k * l_M
        + k
        + 2 * max_l_u
        + 3 * r
        + 1
    )


def uniform_q_resources(n: int, r: int, log2_L_j: float = 0.0) -> dict[str, int]:
    """Evaluate one ``Q`` circuit when every Babai row has the same ``L_j``.

    This algebraically grouped evaluator is used only for the
    data-independent upper-bound sweep (where the appendix sets ``L_j=1``).
    It is exactly equivalent to calling both detailed modules with a length-
    ``k`` constant list, but runs in constant time per ``r``.
    """

    m = n
    f, eta_1, g_eta = QTC.kyber_instance_parameters(n)
    k = QTC.compute_k(n, r)
    l_q = QTC.KYBER_QUBIT_LENGTH
    l_g = QTC.KYBER_GUESS_LENGTH
    l_cw = l_q + l_g - 1
    l_r = QTC.compute_l_r(r)
    l_k = QTC.compute_l_k(k)
    l_t = QTC.compute_l_t(l_r)
    l_tr = QTC.compute_l_tr(l_r)
    l_p = QTC.compute_l_p(l_t, f, r, k)
    l_tilde_b = QTC.compute_l_tilde_b(l_p)
    l_t_tilde_b = QTC.compute_l_t_tilde_b(l_t, l_tilde_b)
    l_M = QTC.compute_l_M(l_t_tilde_b, l_k)
    l_u = QTC.compute_l_u_j(l_t, l_q, l_k, log2_L_j)
    l_L = QTC.compute_l_L_j(l_p, log2_L_j)
    if l_u + l_L != l_M:
        raise AssertionError("fixed-point partition is inconsistent")

    # Toffoli resources.
    cw_td = QTC.compute_cw_g_toffoli_depth(k, l_t, g_eta)
    cw_tc = QTC.compute_cw_g_toffoli_cost(k, l_t, g_eta, r)
    mj_td = QTC.Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k)
    mj_tc = QTC.Mj_toffoli_cost(l_t, l_tilde_b, l_t_tilde_b, k, l_k)
    u_td = QTC.compute_u_j_toffoli_depth(l_M, l_L, l_u)
    u_tc = QTC.compute_u_j_toffoli_cost(l_M, l_L, l_u)
    red_td = QTC.REDj_toffoli_depth(l_u, l_q, l_tr)
    red_tc = QTC.REDj_toffoli_cost(l_u, l_q, l_tr, k)
    np_td = u_td + k * (mj_td + u_td + red_td)
    np_tc = k * (mj_tc + 2 * u_tc + red_tc)
    range_td = QTC.rangecheck_toffoli_depth(l_tr, k)
    range_tc = QTC.rangecheck_toffoli_cost(l_tr, k)
    lwe_td = QTC.compute_lwecheck_toffoli_depth(m, n, l_q, g_eta)
    lwe_tc = QTC.compute_lwecheck_toffoli_cost(m, n, l_q, g_eta)
    s_chi_td = 2 * (cw_td + np_td + range_td + lwe_td) + 1
    s_chi_tc = 2 * (cw_tc + np_tc + range_tc + lwe_tc) + 1
    N = r * l_g
    q_td = s_chi_td + 2 * QTC.ceil_log2(N - 1) - 1
    q_tc = s_chi_tc + 2 * N - 5

    # Total logical-gate resources, using the paper's data-independent bounds.
    E_C = (r + 1) * QTC.ceil_log2(r + 1) + l_g - 1
    e_C = QGC.phi(l_g + QTC.ceil_log2(r + 1) - 1)
    X_C = (k + 1) * l_q
    qrom_cw = QGC.kyber_qrom_pair_gate_bound(eta_1, l_cw)
    cw_gc = (
        k * r * qrom_cw
        + k * (r + 1) * QGC.addition_gate_count(l_t)
        + 2 * k * E_C
        + X_C
    )
    cw_gd = k * (qrom_cw + QGC.addition_gate_depth(l_t) + 2 * e_C)
    mj_gc = (
        2 * k * QGC.negative_constant_multiplication_gate_count(l_t, l_tilde_b)
        + QGC.sum_gate_count(k, l_t_tilde_b)
    )
    mj_gd = (
        2 * QGC.negative_constant_multiplication_gate_depth(l_t, l_tilde_b)
        + QGC.sum_gate_depth(k, l_t_tilde_b)
    )
    u_gc = (
        2 * QGC.constant_division_gate_count(l_M + 1, l_L)
        + QGC.controlled_constant_addition_gate_count(l_u, constant=1)
        + QGC.controlled_constant_addition_gate_count(l_u, constant=-1)
        + l_u
        + 6
    )
    u_gd = (
        2 * QGC.constant_division_gate_depth(l_M + 1, l_L)
        + QGC.controlled_constant_addition_gate_depth(l_u, constant=1)
        + QGC.controlled_constant_addition_gate_depth(l_u, constant=-1)
        + 7
    )
    extension_width = max(l_tr - (l_u + l_q), 0)
    E_R = k * extension_width
    e_R = QGC.phi(extension_width)
    red_gc = (
        2 * k * QGC.negative_constant_multiplication_gate_count(l_u, l_q)
        + k * QGC.addition_gate_count(l_tr)
        + 2 * E_R
        + 2 * l_u * (k - 1)
    )
    red_gd = (
        2 * QGC.negative_constant_multiplication_gate_depth(l_u, l_q)
        + QGC.addition_gate_depth(l_tr)
        + 2 * e_R
        + 2 * QTC.ceil_log2(k)
    )
    np_gc = k * l_tr + k * (mj_gc + 2 * u_gc + red_gc)
    np_gd = QGC.phi(l_tr - l_t + 1) + u_gd + k * (mj_gd + u_gd + red_gd)
    delta_eta = 1 if eta_1 == 2 else 0
    range_gc = (
        k
        * (
            2 * QGC.constant_addition_gate_count(l_tr, constant=eta_1)
            + 2 * l_tr
            - 6
            + delta_eta
        )
        + 2 * k
        - 3
    )
    range_gd = (
        2 * QGC.constant_addition_gate_depth(l_tr)
        + max(
            2 * QTC.ceil_log2(l_tr - 3) - 1,
            2 + delta_eta,
        )
        + 2 * QTC.ceil_log2(k)
        - 1
    )
    qrom_as = QGC.kyber_qrom_pair_gate_bound(eta_1, l_q)
    as_gc = m * n * (qrom_as + QGC.modular_addition_gate_count(l_q))
    as_gd = m * (qrom_as + QGC.modular_addition_gate_depth(l_q))
    check0_gc = 2 * m * l_q - 3
    check0_gd = 2 * QTC.ceil_log2(m * l_q) - 1
    lwe_gc = (
        as_gc
        + m * QGC.modular_addition_gate_count(l_q)
        + m * QGC.constant_modular_addition_gate_count(l_q)
        + check0_gc
    )
    lwe_gd = (
        as_gd
        + QGC.modular_addition_gate_depth(l_q)
        + QGC.constant_modular_addition_gate_depth(l_q)
        + check0_gd
    )
    s_chi_gc = 2 * (cw_gc + np_gc + range_gc + lwe_gc) + 1
    s_chi_gd = 2 * (cw_gd + np_gd + range_gd + lwe_gd) + 1
    q_gc = s_chi_gc + 26 * r + 4 * N - 3
    q_gd = s_chi_gd + 22 + 2 * QTC.ceil_log2(N - 1) + 3

    return {
        "qubits": total_qubit_upper_bound(n, r),
        "toffoli_depth": q_td,
        "toffoli_count": q_tc,
        "gate_depth": q_gd,
        "gate_count": q_gc,
    }


def q_circuit_upper_bounds(
    n: int, *, log2_L_j: float = 0.0
) -> dict[str, dict[str, float | int]]:
    """Maximize each one-``Q`` resource over ``1 <= r < n``."""

    maxima: dict[str, dict[str, float | int]] = {}
    for r in range(1, n):
        resources = uniform_q_resources(n, r, log2_L_j)
        for metric, value in resources.items():
            if metric not in maxima or value > int(maxima[metric]["value"]):
                maxima[metric] = {
                    "value": value,
                    "log2": math.log2(value),
                    "r": r,
                }
    return maxima


def add_resources_at_optimum(candidate: dict, *, q: int = DEFAULT_Q) -> dict:
    """Evaluate TD, TC, GD, and GC once at an objective optimum."""

    if not candidate.get("valid", True):
        raise ValueError("cannot finalize an invalid candidate")
    n, r = int(candidate["n"]), int(candidate["r"])
    profile = _profile_at_candidate(candidate, q)
    log2_L_j_list = (2.0 * profile).tolist()

    resources = QGC.compute_qsearch_resources_log2(
        n, r, log2_L_j_list
    )
    detailed_toffoli = QTC.compute_all(n, r, log2_L_j_list)
    fast_depth = QTC.compute_q_toffoli_depth_fast(n, r, log2_L_j_list)
    if detailed_toffoli["Q"]["depth"] != fast_depth:
        raise AssertionError("fast and detailed Toffoli depths disagree")
    if not math.isclose(
        resources["toffoli_depth"],
        float(candidate["T_hyb_log2"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("search and final Toffoli depths disagree")

    prob_log2 = float(candidate["prob_log2"])
    adjusted = {key: value - prob_log2 for key, value in resources.items()}

    # Clifford+T normalization used by the current NIST AES comparison.
    # This is evaluated only at the selected optimum, so it does not slow the
    # exhaustive (r,beta) objective search.
    ft = QFT.compute_ft_resources_from_log2(n, r, resources)
    ft_per_success = QFT.add_success_probability(ft, prob_log2)
    maxdepth_rows = QFT.maxdepth_comparison(n, ft, prob_log2)
    maxdepth_fields: dict[str, float | int | None] = {}
    for row in maxdepth_rows:
        h = int(row["h"])
        maxdepth_fields[f"FT_MAXDEPTH_h{h}_gate_count_log2"] = float(
            row["FT_MAXDEPTH_gate_count_log2"]
        )
        maxdepth_fields[f"NIST_AES_reference_h{h}_log2"] = row[
            "NIST_AES_reference_log2"
        ]

    qubits = total_qubit_upper_bound(n, r)
    return {
        **candidate,
        "n": n,
        "r": r,
        "block_size": int(candidate["block_size"]),
        "gsa_tail_dimension": int(candidate["gsa_tail_dimension"]),
        "qubit_upper_bound": qubits,
        "qubit_upper_bound_log2": math.log2(qubits),
        "QSearch_Toffoli_depth_log2": resources["toffoli_depth"],
        "QSearch_Toffoli_count_log2": resources["toffoli_count"],
        "QSearch_Gate_depth_log2": resources["gate_depth"],
        "QSearch_Gate_count_log2": resources["gate_count"],
        "Hybrid_Toffoli_depth_per_success_log2": adjusted["toffoli_depth"],
        "Hybrid_Toffoli_count_per_success_log2": adjusted["toffoli_count"],
        "Hybrid_Gate_depth_per_success_log2": adjusted["gate_depth"],
        "Hybrid_Gate_count_per_success_log2": adjusted["gate_count"],
        # Clifford+T normalization and rotation-synthesis budget for the
        # direct NIST AES comparison in the revised Section 7.4.
        **ft,
        **ft_per_success,
        **maxdepth_fields,
        # Compatibility with older tables in the project.
        "Toffoli_cost_log2_hyp": adjusted["toffoli_count"],
    }


def _load_checkpoint(path: Path, n: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"model_version", "n", "r", "valid"}
    if not required.issubset(frame.columns):
        raise ValueError(f"incompatible checkpoint schema: {path}")
    versions = set(frame["model_version"].dropna().astype(str))
    if versions != {MODEL_VERSION}:
        raise ValueError(
            f"checkpoint {path} belongs to another formula version; "
            "use a new output directory"
        )
    if set(frame["n"].astype(int)) != {n}:
        raise ValueError(f"checkpoint n does not match {n}: {path}")
    return frame


def _save_checkpoint(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = (
        frame.drop_duplicates(subset="r", keep="last")
        .sort_values("r")
        .reset_index(drop=True)
    )
    frame.to_csv(path, index=False)
    return frame


def search_best_for_n(
    n: int,
    *,
    q: int = DEFAULT_Q,
    output_dir: str | os.PathLike[str] = "section7_results",
    workers: int = 1,
    r_values: Iterable[int] | None = None,
) -> dict:
    """Run the exhaustive, resumable search and finalize the global optimum."""

    if n not in QTC.SUPPORTED_DIMENSIONS:
        raise ValueError(f"unsupported n={n}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = output_path / f"objective_search_results_{n}.csv"
    final_path = output_path / f"best_cost_log2_results_{n}.csv"
    frame = _load_checkpoint(checkpoint, n)
    completed = set(frame["r"].astype(int)) if not frame.empty else set()
    requested = list(range(1, n)) if r_values is None else list(r_values)
    todo = [r for r in requested if 1 <= r < n and r not in completed]

    if workers <= 1:
        for index, r in enumerate(todo, start=1):
            row = search_best_objective_for_r(n, r, q=q)
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            frame = _save_checkpoint(frame, checkpoint)
            print(f"n={n}: completed r={r} ({index}/{len(todo)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_search_task, (n, r, q)): r for r in todo
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                frame = pd.concat(
                    [frame, pd.DataFrame([row])], ignore_index=True
                )
                frame = _save_checkpoint(frame, checkpoint)
                print(
                    f"n={n}: completed r={futures[future]} "
                    f"({index}/{len(todo)})",
                    flush=True,
                )

    valid = frame[frame["valid"].astype(bool)]
    if valid.empty:
        raise RuntimeError(f"no valid candidate found for n={n}")
    best_row = valid.loc[valid["total_cost_log2"].idxmin()].to_dict()
    optimum = add_resources_at_optimum(best_row, q=q)
    pd.DataFrame([optimum]).to_csv(final_path, index=False)
    with (output_path / f"best_cost_log2_results_{n}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(optimum, handle, indent=2, sort_keys=True)
    return optimum


def write_section7_ft_tables(
    results: Mapping[int, Mapping[str, object]],
    output_dir: str | os.PathLike[str] = "section7_results",
) -> None:
    """Write the three new Section 7.4 FT/NIST tables as CSV files.

    The files are lightweight post-processing products of the selected optima:
    ``rotation_synthesis_budget.csv``, ``ft_gate_resources.csv``, and
    ``maxdepth_gate_comparison.csv``.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rotation_rows = []
    ft_rows = []
    maxdepth_rows = []
    for n in sorted(results):
        result = results[n]
        label = QFT.instance_label(n)
        rotation_rows.append(
            {
                "Parameter Set": label,
                "n": n,
                "r": int(result["r"]),
                "log2_L": float(result["log2_L_rotation_budget"]),
                "log2_N_R": float(result["rotation_count_log2"]),
                "b_rot": int(result["rotation_precision_bits"]),
                "T_R_per_rotation": int(result["rotation_T_per_gate"]),
                "log2_N_R_T_R": float(result["rotation_T_count_log2"]),
            }
        )
        ft_rows.append(
            {
                "Parameter Set": label,
                "n": n,
                "r": int(result["r"]),
                "log2_D_FT": float(result["QSearch_FT_depth_log2"]),
                "log2_G_FT": float(result["QSearch_FT_gate_count_log2"]),
                "log2_T_FT": float(result["QSearch_FT_T_count_log2"]),
                "log2_D_FT_per_success": float(
                    result["Hybrid_FT_depth_per_success_log2"]
                ),
                "log2_G_FT_per_success": float(
                    result["Hybrid_FT_gate_count_per_success_log2"]
                ),
                "log2_T_FT_per_success": float(
                    result["Hybrid_FT_T_count_per_success_log2"]
                ),
            }
        )
        for h in QFT.MAXDEPTH_EXPONENTS:
            maxdepth_rows.append(
                {
                    "Parameter Set": label,
                    "n": n,
                    "h": h,
                    "log2_G_MD_FT": float(
                        result[f"FT_MAXDEPTH_h{h}_gate_count_log2"]
                    ),
                    "NIST_AES_reference_log2": result.get(
                        f"NIST_AES_reference_h{h}_log2"
                    ),
                }
            )

    pd.DataFrame(rotation_rows).to_csv(
        output_path / "rotation_synthesis_budget.csv", index=False
    )
    pd.DataFrame(ft_rows).to_csv(
        output_path / "ft_gate_resources.csv", index=False
    )
    pd.DataFrame(maxdepth_rows).to_csv(
        output_path / "maxdepth_gate_comparison.csv", index=False
    )


def run_cost_estimation(
    n_list: Sequence[int] = (256, 512, 768, 1024),
    *,
    q: int = DEFAULT_Q,
    output_dir: str | os.PathLike[str] = "section7_results",
    workers: int = 1,
) -> dict[int, dict]:
    """Recompute the global optimum and all four quantum metrics."""

    results: dict[int, dict] = {}
    upper_bounds: dict[int, dict[str, dict[str, float | int]]] = {}
    for n in n_list:
        results[n] = search_best_for_n(
            n, q=q, output_dir=output_dir, workers=workers
        )
        upper_bounds[n] = q_circuit_upper_bounds(n)
        print(
            f"n={n}: optimum r={results[n]['r']}, "
            f"beta={results[n]['block_size']}, "
            f"log2 total={results[n]['total_cost_log2']:.6f}",
            flush=True,
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "q_circuit_upper_bounds.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(upper_bounds, handle, indent=2, sort_keys=True)
    write_section7_ft_tables(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024],
        help="flattened LWE dimensions to evaluate",
    )
    parser.add_argument("--q", type=int, default=DEFAULT_Q)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="section7_results")
    args = parser.parse_args()
    run_cost_estimation(
        tuple(args.n),
        q=args.q,
        output_dir=args.output_dir,
        workers=max(1, args.workers),
    )


if __name__ == "__main__":
    main()
