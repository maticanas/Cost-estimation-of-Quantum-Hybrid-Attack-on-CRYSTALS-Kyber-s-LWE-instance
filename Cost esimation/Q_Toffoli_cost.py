"""Toffoli resources for the Kyber quantum-hybrid-attack circuit.

The formulas in this module follow Sections 3, 5, and 6 of the revised
manuscript.  ``cost`` is retained as a compatibility alias for Toffoli count.
All values returned by :func:`compute_all` are for one execution of the named
circuit; the expected number of Grover iterations is handled separately in
logarithmic form by :func:`compute_qsearch_toffoli_log2`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Mapping, Sequence


SUPPORTED_DIMENSIONS = (256, 512, 768, 1024)
KYBER_QUBIT_LENGTH = 12
KYBER_GUESS_LENGTH = 3


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------


def ceil(x: float) -> int:
    """Return ``ceil(x)`` as an integer."""

    return math.ceil(x)


def ceil_log2(x: float) -> int:
    """Return ``ceil(log2(x))`` for a positive number."""

    if x <= 0:
        raise ValueError("ceil_log2 requires x > 0")
    return math.ceil(math.log2(x))


def log2_rounding_term(k: int, log2_L: float) -> float:
    r"""Compute ``log2(1 - 2**(-1/(kL)))`` without constructing ``L``.

    The manuscript uses ``L = 2**(f*r)``.  Directly evaluating
    ``1 - 2**(-1/(k*L))`` loses all precision for realistic parameters.  Put
    ``z = ln(2)/(kL)`` and evaluate ``log2(1-exp(-z))`` with ``expm1``.  When
    ``z`` is below floating-point range, ``1-exp(-z) = z + O(z**2)`` is used.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    log2_z = math.log2(math.log(2.0)) - math.log2(k) - log2_L
    if log2_z < -50.0:
        # The relative correction is below 2^-51, so it cannot affect the
        # represented double except at an exact integer boundary.
        return log2_z

    z = math.exp2(log2_z)
    return math.log2(-math.expm1(-z))


def kyber_instance_parameters(n: int) -> tuple[float, int, int]:
    """Return ``(f, eta_1, g(eta_1))`` for the selected parameter set.

    ``n=256`` is retained as the non-standard test case used by the original
    notebook.  Here ``g(eta_1)`` is the combined Toffoli cost of one optimized
    QROM lookup and its inverse.
    """

    if n in (256, 512):
        return 1.302, 3, 20
    if n in (768, 1024):
        return 1.108, 2, 16
    raise ValueError(f"n must be one of {SUPPORTED_DIMENSIONS}")


# ---------------------------------------------------------------------------
# Length parameters
# ---------------------------------------------------------------------------


def compute_l_r(r: int) -> int:
    """Compute ``l_r = ceil(log2(r + 1))``."""

    return ceil_log2(r + 1)


def compute_k(n: int, r: int) -> int:
    """Compute ``k = 2n - r``."""

    if n <= 0 or not 1 <= r <= n:
        raise ValueError("require n > 0 and 1 <= r <= n")
    return 2 * n - r


def compute_l_k(k: int) -> int:
    return ceil_log2(k)


def compute_l_t(l_r: int) -> int:
    """Compute ``l_t = 14 + l_r`` for Kyber."""

    return 14 + l_r


def compute_l_tr(l_r: int) -> int:
    """Compute ``l_tr = 26 + l_r`` for Kyber."""

    return 26 + l_r


def compute_l_p(l_t: int, f: float, r: int, k: int) -> int:
    r"""Compute the precision length in Table 6 of the manuscript.

    .. math::
       l_p=l_t-1+\left\lceil\log_2 k-
       \log_2\left(1-2^{-1/(kL)}\right)\right\rceil,
       \qquad L=2^{fr}.
    """

    log2_term = log2_rounding_term(k, f * r)
    return (l_t - 1) + math.ceil(math.log2(k) - log2_term)


def compute_l_tilde_b(l_p: int) -> int:
    """Compute ``l_tilde_b = l_q + l_p = 12 + l_p``."""

    return KYBER_QUBIT_LENGTH + l_p


def compute_l_t_tilde_b(l_t: int, l_tilde_b: int) -> int:
    return l_t + l_tilde_b


def compute_l_M(l_t_tilde_b: int, l_k: int) -> int:
    return l_t_tilde_b + l_k


def compute_l_u_j(l_t: int, l_q: int, l_k: int, log2_L_j: float) -> int:
    r"""Compute the quotient-register length used after rounding.

    With ``l_L_j=floor(log2(L_j))+l_p+1`` and
    ``l_M=l_t+l_q+l_p+l_k``, the manuscript partition is

    ``l_u_j = l_M-l_L_j
              = l_t+l_q+l_k-floor(log2(L_j))-1``.

    The constant-division input has length ``l_M+1``; hence its raw signed
    quotient has length ``(l_M+1)-l_L_j = l_u_j+1``.  The extra bit is the
    precision/rounding bit shown explicitly in the circuit.
    """

    result = l_t + l_q + l_k - math.floor(log2_L_j) - 1
    if result <= 0:
        raise ValueError("computed l_u_j is not positive")
    return result


def compute_l_L_j(l_p: int, log2_L_j: float) -> int:
    r"""Compute ``l_L_j = floor(log2(L_j)) + l_p + 1``."""

    result = math.floor(log2_L_j) + l_p + 1
    if result <= 0:
        raise ValueError("computed l_L_j is not positive")
    return result


# ---------------------------------------------------------------------------
# Toffoli costs of arithmetic primitives (Section 3.1)
# ---------------------------------------------------------------------------


def table_lookup_cost(w: int, n: int) -> int:
    """Toffoli count/depth of ``QROM(w,n)``."""

    del n  # Output width affects logical CNOTs, but not the Toffoli metric.
    return 2 ** (w + 1) - 4


def addition_cost(n: int) -> int:
    return 2 * n - 2


def constant_addition_cost(n: int) -> int:
    return 2 * n - 4


def controlled_constant_addition_cost(n: int) -> int:
    return 2 * n - 2


def modular_addition_cost(n: int) -> int:
    return 8 * n - 7


def constant_modular_addition_cost(n: int) -> int:
    return 8 * n - 9


def unsigned_product_addition_cost(n: int, m: int) -> int:
    """Toffoli cost of ``PA_c(n,m)`` with ``w=ceil(log2(n))``."""

    w = ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_cost(w, m + w)
        + (s - 1) * addition_cost(m + w)
    )


def positive_constant_multiplication_cost(n: int, m: int) -> int:
    return unsigned_product_addition_cost(n - 1, m) + 2 * (n + m) - 2


def negative_constant_multiplication_cost(n: int, m: int) -> int:
    return unsigned_product_addition_cost(n - 1, m) + 4 * (n + m) - 4


def unsigned_modular_product_addition_cost(n: int, m: int) -> int:
    """Toffoli cost of ``PA_c^q(n,m)``."""

    del m
    w = ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_cost(w, n)
        + (s - 1) * modular_addition_cost(n)
    )


def constant_modular_multiplication_cost(n: int, m: int) -> int:
    return unsigned_modular_product_addition_cost(n, m)


def constant_division_cost(n: int, m: int) -> int:
    r"""Toffoli count/depth of ``D_c(n,m)`` in the manuscript.

    ``n`` is the full signed-dividend register length, ``m`` is the unsigned
    remainder-register length, and the signed quotient occupies ``n-m``
    qubits.  This is intentionally *not* the alternative parameterization
    whose polynomial starts with ``2*m**2``.
    """

    if n <= 0 or m <= 0 or m >= n:
        raise ValueError("constant division requires 0 < m < n")
    return 2 * n**2 - 2 * m**2 - 6 * n + 4 * m + 3


# ---------------------------------------------------------------------------
# Toffoli costs of component circuits (Sections 5 and 6)
# ---------------------------------------------------------------------------


def compute_cw_g_toffoli_depth(k: int, l_t: int, g_eta: int) -> int:
    return k * (addition_cost(l_t) + g_eta)


def compute_cw_g_toffoli_cost(k: int, l_t: int, g_eta: int, r: int) -> int:
    return k * (r + 1) * addition_cost(l_t) + g_eta * k * r


def Mj_toffoli_depth(l_t: int, l_tilde_b: int, l_t_tilde_b: int, l_k: int) -> int:
    sum_depth = 2 * sum(
        addition_cost(l_t_tilde_b + i) for i in range(1, l_k + 1)
    )
    return 2 * negative_constant_multiplication_cost(l_t, l_tilde_b) + sum_depth


def Mj_toffoli_cost(
    l_t: int,
    l_tilde_b: int,
    l_t_tilde_b: int,
    k: int,
    l_k: int,
) -> int:
    sum_cost = 2 * sum(
        math.ceil(k / (2**i)) * addition_cost(l_t_tilde_b + i)
        for i in range(1, l_k + 1)
    )
    return 2 * k * negative_constant_multiplication_cost(l_t, l_tilde_b) + sum_cost


def compute_u_j_toffoli_depth(l_M: int, l_L_j: int, l_u_j: int) -> int:
    """Toffoli depth of ``u_j``, including the two controlled rounding boxes."""

    return (
        2 * constant_division_cost(l_M + 1, l_L_j)
        + controlled_constant_addition_cost(l_u_j)
        + controlled_constant_addition_cost(l_u_j)
        + 4
    )


def compute_u_j_toffoli_cost(l_M: int, l_L_j: int, l_u_j: int) -> int:
    return compute_u_j_toffoli_depth(l_M, l_L_j, l_u_j)


def REDj_toffoli_depth(l_u_j: int, l_b_j_x: int, l_tr: int) -> int:
    return 2 * negative_constant_multiplication_cost(l_u_j, l_b_j_x) + addition_cost(l_tr)


def REDj_toffoli_cost(l_u_j: int, l_b_j_x: int, l_tr: int, k: int) -> int:
    return (
        2 * k * negative_constant_multiplication_cost(l_u_j, l_b_j_x)
        + k * addition_cost(l_tr)
    )


def rangecheck_toffoli_depth(l_tr: int, k: int) -> int:
    comparison_depth = max(2 * ceil_log2(l_tr - 3) - 1, 2)
    final_mpmct_depth = 2 * ceil_log2(k) - 1
    return 2 * constant_addition_cost(l_tr) + comparison_depth + final_mpmct_depth


def rangecheck_toffoli_cost(l_tr: int, k: int) -> int:
    return 2 * k * constant_addition_cost(l_tr) + 2 * k * l_tr - 4 * k - 3


def compute_lwecheck_toffoli_depth(m: int, n: int, l_q: int, g_eta: int) -> int:
    """Toffoli depth of the optimized Kyber ``LWEcheck`` circuit."""

    del n
    return (
        (m + 1) * modular_addition_cost(l_q)
        + constant_modular_addition_cost(l_q)
        + 2 * ceil_log2(m * l_q)
        + g_eta * m
        - 1
    )


def compute_lwecheck_toffoli_cost(m: int, n: int, l_q: int, g_eta: int) -> int:
    """Toffoli count of the optimized Kyber ``LWEcheck`` circuit."""

    return (
        (m * n + m) * modular_addition_cost(l_q)
        + m * constant_modular_addition_cost(l_q)
        + 2 * m * l_q
        + g_eta * m * n
        - 3
    )


def _metric(count: int, depth: int) -> dict[str, int]:
    """Build a result record while preserving the old ``cost`` key."""

    return {"count": count, "cost": count, "depth": depth}


def compute_all(n: int, r: int, log2_L_j_list: Sequence[float]) -> dict:
    """Compute all Kyber-specific Toffoli resources for one ``Q`` iteration.

    ``log2_L_j_list`` must contain one value for every Babai iteration, i.e.
    exactly ``k = 2n-r`` entries in the same order as ``j=1,...,k``.
    """

    m = n
    f, eta_1, g_eta = kyber_instance_parameters(n)
    k = compute_k(n, r)
    if len(log2_L_j_list) != k:
        raise ValueError(
            f"log2_L_j_list must have k={k} entries; got {len(log2_L_j_list)}"
        )

    l_q = KYBER_QUBIT_LENGTH
    l_g = KYBER_GUESS_LENGTH
    l_cw = l_q + l_g - 1
    l_b_j_x = l_q
    l_r = compute_l_r(r)
    l_t = compute_l_t(l_r)
    l_tr = compute_l_tr(l_r)
    l_p = compute_l_p(l_t, f, r, k)
    l_tilde_b = compute_l_tilde_b(l_p)
    l_t_tilde_b = compute_l_t_tilde_b(l_t, l_tilde_b)
    l_k = compute_l_k(k)
    l_M = compute_l_M(l_t_tilde_b, l_k)

    rangecheck_count = rangecheck_toffoli_cost(l_tr, k)
    rangecheck_depth = rangecheck_toffoli_depth(l_tr, k)
    mj_count = Mj_toffoli_cost(l_t, l_tilde_b, l_t_tilde_b, k, l_k)
    mj_depth = Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k)
    lwecheck_count = compute_lwecheck_toffoli_cost(m, n, l_q, g_eta)
    lwecheck_depth = compute_lwecheck_toffoli_depth(m, n, l_q, g_eta)
    cw_count = compute_cw_g_toffoli_cost(k, l_t, g_eta, r)
    cw_depth = compute_cw_g_toffoli_depth(k, l_t, g_eta)

    per_j = []
    np_b_count = 0
    pipelined_depth_sum = 0
    for j, log2_L_j in enumerate(log2_L_j_list, start=1):
        l_u_j = compute_l_u_j(l_t, l_q, l_k, log2_L_j)
        l_L_j = compute_l_L_j(l_p, log2_L_j)
        if l_L_j + l_u_j != l_M:
            raise AssertionError("fixed-point partition must satisfy l_L_j+l_u_j=l_M")
        if (l_M + 1) - l_L_j != l_u_j + 1:
            raise AssertionError("division quotient must contain l_u_j+1 bits")
        u_count = compute_u_j_toffoli_cost(l_M, l_L_j, l_u_j)
        u_depth = compute_u_j_toffoli_depth(l_M, l_L_j, l_u_j)
        red_count = REDj_toffoli_cost(l_u_j, l_b_j_x, l_tr, k)
        red_depth = REDj_toffoli_depth(l_u_j, l_b_j_x, l_tr)
        np_b_j_count = mj_count + 2 * u_count + red_count
        np_b_j_depth = mj_depth + 2 * u_depth + red_depth

        np_b_count += np_b_j_count
        pipelined_depth_sum += mj_depth + u_depth + red_depth
        per_j.append(
            {
                "j": j,
                "log2_L_j": log2_L_j,
                "l_u_j": l_u_j,
                "l_L_j": l_L_j,
                "u_j": _metric(u_count, u_depth),
                "REDj": _metric(red_count, red_depth),
                "NP_B_j": _metric(np_b_j_count, np_b_j_depth),
            }
        )

    # Pipeline formula: TD(u_1) + sum_j[TD(M_j)+TD(u_j)+TD(RED_j)].
    np_b_depth = per_j[0]["u_j"]["depth"] + pipelined_depth_sum

    s_chi_count = (
        2 * cw_count
        + 2 * np_b_count
        + 2 * lwecheck_count
        + 2 * rangecheck_count
        + 1
    )
    s_chi_depth = (
        2 * cw_depth
        + 2 * np_b_depth
        + 2 * lwecheck_depth
        + 2 * rangecheck_depth
        + 1
    )

    # T(P) has no Toffoli gates.  S_0 has N-1 controls for N=r*l_g.
    N = r * l_g
    s0_count = 2 * N - 5
    s0_depth = 2 * ceil_log2(N - 1) - 1
    q_count = s_chi_count + s0_count
    q_depth = s_chi_depth + s0_depth

    parameters = {
        "n": n,
        "m": m,
        "r": r,
        "k": k,
        "f": f,
        "eta_1": eta_1,
        "g_eta": g_eta,
        "log2_L": f * r,
        "l_q": l_q,
        "l_g": l_g,
        "l_cw": l_cw,
        "l_b_j_x": l_b_j_x,
        "l_r": l_r,
        "l_k": l_k,
        "l_t": l_t,
        "l_tr": l_tr,
        "l_p": l_p,
        "l_tilde_b": l_tilde_b,
        "l_t_tilde_b": l_t_tilde_b,
        "l_M": l_M,
    }

    return {
        "parameters": parameters,
        "T_P": _metric(0, 0),
        "Cw_g": _metric(cw_count, cw_depth),
        "Mj": _metric(mj_count, mj_depth),
        # Compatibility: these two keys refer to the final j.  Use ``per_j``
        # for the complete, non-uniform list.
        "u_j": per_j[-1]["u_j"],
        "REDj": per_j[-1]["REDj"],
        "per_j": per_j,
        "NP_B": _metric(np_b_count, np_b_depth),
        "RangeCheck": _metric(rangecheck_count, rangecheck_depth),
        "LWECheck": _metric(lwecheck_count, lwecheck_depth),
        "S_chi": _metric(s_chi_count, s_chi_depth),
        "S_0": _metric(s0_count, s0_depth),
        "Q": _metric(q_count, q_depth),
    }


def compute_toffoli_depth_and_count(
    n: int, r: int, log2_L_j_list: Sequence[float]
) -> tuple[float, float]:
    """Return ``(log2 TD(Q), log2 TC(Q))`` for one ``Q`` iteration."""

    results = compute_all(n, r, log2_L_j_list)
    return math.log2(results["Q"]["depth"]), math.log2(results["Q"]["count"])


def compute_q_toffoli_depth_fast(
    n: int, r: int, log2_L_j_list: Sequence[float]
) -> int:
    """Return ``TD(Q)`` without materializing all per-``j`` records.

    The exhaustive Section 7 search calls this function for many candidate
    bases.  At the Toffoli-depth level, a Babai row depends on ``L_j`` only
    through ``floor(log2(L_j))``.  Grouping equal floors preserves the exact
    manuscript formula while avoiding thousands of repeated Python calls.
    ``compute_all`` remains the authoritative detailed evaluator and is used
    once at the selected optimum.
    """

    k = compute_k(n, r)
    if len(log2_L_j_list) != k:
        raise ValueError(
            f"log2_L_j_list must have k={k} entries; got {len(log2_L_j_list)}"
        )

    floor_values = [math.floor(value) for value in log2_L_j_list]
    return compute_q_toffoli_depth_grouped(
        n,
        r,
        first_floor=floor_values[0],
        floor_multiplicities=Counter(floor_values),
    )


def compute_q_toffoli_depth_grouped(
    n: int,
    r: int,
    *,
    first_floor: int,
    floor_multiplicities: Mapping[int, int],
) -> int:
    """Return ``TD(Q)`` from multiplicities of ``floor(log2(L_j))``.

    This interface lets the numerical search pass the run-length structure of
    the modified GSA profile directly.  The multiplicities must sum to
    ``k=2n-r``; ``first_floor`` identifies the value for the extra pipelined
    ``u_1`` term.
    """

    m = n
    f, _eta_1, g_eta = kyber_instance_parameters(n)
    k = compute_k(n, r)
    if sum(floor_multiplicities.values()) != k:
        raise ValueError("floor multiplicities must sum to k=2n-r")
    if first_floor not in floor_multiplicities:
        raise ValueError("first_floor must occur in floor_multiplicities")

    l_q = KYBER_QUBIT_LENGTH
    l_g = KYBER_GUESS_LENGTH
    l_r = compute_l_r(r)
    l_k = compute_l_k(k)
    l_t = compute_l_t(l_r)
    l_tr = compute_l_tr(l_r)
    l_p = compute_l_p(l_t, f, r, k)
    l_tilde_b = compute_l_tilde_b(l_p)
    l_t_tilde_b = compute_l_t_tilde_b(l_t, l_tilde_b)
    l_M = compute_l_M(l_t_tilde_b, l_k)

    mj_depth = Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k)
    range_depth = rangecheck_toffoli_depth(l_tr, k)
    lwe_depth = compute_lwecheck_toffoli_depth(m, n, l_q, g_eta)
    cw_depth = compute_cw_g_toffoli_depth(k, l_t, g_eta)

    def row_depth(floor_log2_L_j: int) -> tuple[int, int]:
        l_u_j = l_t + l_q + l_k - floor_log2_L_j - 1
        l_L_j = floor_log2_L_j + l_p + 1
        if l_u_j <= 0 or l_L_j <= 0 or l_u_j + l_L_j != l_M:
            raise ValueError("invalid fixed-point lengths for a Babai row")
        u_depth = compute_u_j_toffoli_depth(l_M, l_L_j, l_u_j)
        red_depth = REDj_toffoli_depth(l_u_j, l_q, l_tr)
        return u_depth, red_depth

    first_u_depth = row_depth(first_floor)[0]
    summed_rows = 0
    for floor_value, multiplicity in floor_multiplicities.items():
        if multiplicity <= 0:
            raise ValueError("floor multiplicities must be positive")
        u_depth, red_depth = row_depth(floor_value)
        summed_rows += multiplicity * (mj_depth + u_depth + red_depth)

    np_b_depth = first_u_depth + summed_rows
    s_chi_depth = (
        2 * cw_depth
        + 2 * np_b_depth
        + 2 * lwe_depth
        + 2 * range_depth
        + 1
    )
    N = r * l_g
    s0_depth = 2 * ceil_log2(N - 1) - 1
    return s_chi_depth + s0_depth


def compute_qsearch_toffoli_depth_log2_fast(
    n: int, r: int, log2_L_j_list: Sequence[float]
) -> float:
    """Fast exact value of ``log2(L*TD(Q))`` for the search objective."""

    q_depth = compute_q_toffoli_depth_fast(n, r, log2_L_j_list)
    return kyber_instance_parameters(n)[0] * r + math.log2(q_depth)


def compute_Toffoli_depth_and_cost(
    n: int, r: int, log2_L_j_list: Sequence[float]
) -> tuple[float, float]:
    """Backward-compatible alias for :func:`compute_toffoli_depth_and_count`."""

    return compute_toffoli_depth_and_count(n, r, log2_L_j_list)


def compute_qsearch_toffoli_log2(
    n: int, r: int, log2_L_j_list: Sequence[float]
) -> tuple[float, float]:
    """Return expected QSearch Toffoli depth/count after ``L=2**(fr)`` runs."""

    q_depth, q_count = compute_toffoli_depth_and_count(n, r, log2_L_j_list)
    log2_L = kyber_instance_parameters(n)[0] * r
    return log2_L + q_depth, log2_L + q_count


def compute_uniform_toffoli_resources(
    n: int, r: int, log2_L_j: float = 0.0
) -> dict[str, dict[str, int]]:
    """Evaluate Toffoli resources when all Babai rows use the same ``L_j``.

    This is algebraically identical to ``compute_all(n,r,[x]*k)`` but groups
    the repeated rows, making the data-independent upper-bound sweep fast.
    """

    m = n
    f, _eta_1, g_eta = kyber_instance_parameters(n)
    k = compute_k(n, r)
    l_q = KYBER_QUBIT_LENGTH
    l_g = KYBER_GUESS_LENGTH
    l_r = compute_l_r(r)
    l_k = compute_l_k(k)
    l_t = compute_l_t(l_r)
    l_tr = compute_l_tr(l_r)
    l_p = compute_l_p(l_t, f, r, k)
    l_tilde_b = compute_l_tilde_b(l_p)
    l_t_tilde_b = compute_l_t_tilde_b(l_t, l_tilde_b)
    l_M = compute_l_M(l_t_tilde_b, l_k)
    l_u = compute_l_u_j(l_t, l_q, l_k, log2_L_j)
    l_L = compute_l_L_j(l_p, log2_L_j)

    cw_depth = compute_cw_g_toffoli_depth(k, l_t, g_eta)
    cw_count = compute_cw_g_toffoli_cost(k, l_t, g_eta, r)
    mj_depth = Mj_toffoli_depth(l_t, l_tilde_b, l_t_tilde_b, l_k)
    mj_count = Mj_toffoli_cost(l_t, l_tilde_b, l_t_tilde_b, k, l_k)
    u_depth = compute_u_j_toffoli_depth(l_M, l_L, l_u)
    u_count = compute_u_j_toffoli_cost(l_M, l_L, l_u)
    red_depth = REDj_toffoli_depth(l_u, l_q, l_tr)
    red_count = REDj_toffoli_cost(l_u, l_q, l_tr, k)
    np_depth = u_depth + k * (mj_depth + u_depth + red_depth)
    np_count = k * (mj_count + 2 * u_count + red_count)
    range_depth = rangecheck_toffoli_depth(l_tr, k)
    range_count = rangecheck_toffoli_cost(l_tr, k)
    lwe_depth = compute_lwecheck_toffoli_depth(m, n, l_q, g_eta)
    lwe_count = compute_lwecheck_toffoli_cost(m, n, l_q, g_eta)
    s_chi_depth = 2 * (cw_depth + np_depth + range_depth + lwe_depth) + 1
    s_chi_count = 2 * (cw_count + np_count + range_count + lwe_count) + 1
    N = r * l_g
    q_depth = s_chi_depth + 2 * ceil_log2(N - 1) - 1
    q_count = s_chi_count + 2 * N - 5
    return {
        "S_chi": _metric(s_chi_count, s_chi_depth),
        "Q": _metric(q_count, q_depth),
    }


def compute_max_depth_costs() -> list[dict]:
    """Sweep the manuscript's ``L_j=1`` one-``Q`` upper bound efficiently."""

    results = []
    for n in (512, 768, 1024):
        maxima = {"S_chi_depth": 0, "S_chi_count": 0, "Q_depth": 0, "Q_count": 0}
        details = {key: {} for key in maxima}
        print(f"Computing for n={n}")
        for r in range(1, n):
            current = compute_uniform_toffoli_resources(n, r, 0.0)
            candidates = {
                "S_chi_depth": current["S_chi"]["depth"],
                "S_chi_count": current["S_chi"]["count"],
                "Q_depth": current["Q"]["depth"],
                "Q_count": current["Q"]["count"],
            }
            for key, value in candidates.items():
                if value > maxima[key]:
                    maxima[key] = value
                    details[key] = {"n": n, "r": r, "log2_L_j": 0.0}
        result = {"n": n, **maxima, "details": details}
        results.append(result)
        print(
            f"  Q depth: {maxima['Q_depth']} "
            f"(log2={math.log2(maxima['Q_depth']):.5f}, {details['Q_depth']})"
        )
        print(
            f"  Q count: {maxima['Q_count']} "
            f"(log2={math.log2(maxima['Q_count']):.5f}, {details['Q_count']})"
        )
    return results


if __name__ == "__main__":
    compute_max_depth_costs()
