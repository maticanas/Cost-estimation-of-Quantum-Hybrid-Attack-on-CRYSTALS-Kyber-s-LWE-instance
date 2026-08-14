"""Logical gate count and gate depth for the revised Kyber attack circuit.

This module implements the ``GC`` and ``GD`` compositions in Sections 3.2,
3.3, 5, and 6 of the revised manuscript.  The main entry point,
:func:`compute_all`, uses the data-independent upper bounds stated in the
paper whenever a classical table or matrix entry is unavailable.  In
particular:

* generic QROM data-loading costs use ``(output_bits+3) * 2**w - 5``;
* optimized Kyber QROM pairs use ``34 + 4*eta_1*output_bits``;
* unknown constant-adder Hamming weights use their width as an upper bound;
* ``X_C`` uses ``(k+1)l_q`` unless an exact value is supplied.

The resulting values are therefore reproducible upper bounds, not averages.
Helpers for exact QROM-pair costs are included for use when matrix entries are
available.
"""

from __future__ import annotations

import math
from typing import Sequence

import Q_Toffoli_cost as QTC


# ---------------------------------------------------------------------------
# Gate-model helpers
# ---------------------------------------------------------------------------


def phi(d: int) -> int:
    r"""Return ``Phi(d) = ceil(log2(d+1))`` for coherent fanout."""

    if d < 0:
        raise ValueError("phi requires d >= 0")
    return QTC.ceil_log2(d + 1)


def twos_complement_hamming_weight(value: int, width: int) -> int:
    """Hamming weight of ``value`` encoded on ``width`` two's-complement bits."""

    if width <= 0:
        raise ValueError("width must be positive")
    return (value & ((1 << width) - 1)).bit_count()


def _resolve_hamming_weight(
    width: int,
    *,
    constant: int | None = None,
    hamming_weight: int | None = None,
) -> int:
    if hamming_weight is not None:
        if not 0 <= hamming_weight <= width:
            raise ValueError("hamming_weight must lie in [0,width]")
        return hamming_weight
    if constant is not None:
        return twos_complement_hamming_weight(constant, width)
    return width


def _log2_sum(x: float, y: float) -> float:
    """Return ``log2(2**x + 2**y)`` stably."""

    high, low = max(x, y), min(x, y)
    return high + math.log2(1.0 + math.exp2(low - high))


def _sum_integers(lo: int, hi: int) -> int:
    if hi < lo:
        return 0
    return (lo + hi) * (hi - lo + 1) // 2


def _phi_prefix(d: int) -> int:
    """Return ``sum(phi(i) for i in range(d+1))`` in O(1)."""

    if d < 0:
        return 0
    q = d + 1
    b = QTC.ceil_log2(q)
    if b == 0:
        return 0
    complete = 0 if b == 1 else (b - 2) * (2 ** (b - 1)) + 1
    return complete + b * (q - 2 ** (b - 1))


def _sum_phi(lo: int, hi: int) -> int:
    return _phi_prefix(hi) - _phi_prefix(lo - 1)


# ---------------------------------------------------------------------------
# Gate count/depth of arithmetic primitives (Sections 3.2 and 3.3)
# ---------------------------------------------------------------------------


def table_lookup_gate_count(
    w: int, output_bits: int, data_hamming_sum: int | None = None
) -> int:
    """Gate count of ``QROM(w,output_bits)``.

    Supply ``data_hamming_sum=H_P`` for the exact value.  If omitted, the
    matrix-independent upper bound ``H_P <= output_bits*2**w`` is used.
    """

    if data_hamming_sum is None:
        return (output_bits + 3) * (2**w) - 5
    if not 0 <= data_hamming_sum <= output_bits * (2**w):
        raise ValueError("invalid QROM data_hamming_sum")
    return 3 * (2**w) - 5 + data_hamming_sum


def table_lookup_gate_depth(
    w: int, output_bits: int, data_hamming_sum: int | None = None
) -> int:
    # Every counted QROM gate lies on the critical path in the adopted model.
    return table_lookup_gate_count(w, output_bits, data_hamming_sum)


def addition_gate_count(n: int) -> int:
    return 7 * n - 8


def addition_gate_depth(n: int) -> int:
    return 5 * n - 5


def constant_addition_gate_count(
    n: int,
    *,
    constant: int | None = None,
    hamming_weight: int | None = None,
) -> int:
    h_c = _resolve_hamming_weight(
        n, constant=constant, hamming_weight=hamming_weight
    )
    return 7 * n - 8 + 2 * h_c


def constant_addition_gate_depth(n: int) -> int:
    return 5 * n - 3


def controlled_constant_addition_gate_count(
    n: int,
    *,
    constant: int | None = None,
    hamming_weight: int | None = None,
) -> int:
    h_c = _resolve_hamming_weight(
        n, constant=constant, hamming_weight=hamming_weight
    )
    return 7 * n - 8 + 2 * h_c


def controlled_constant_addition_gate_depth(
    n: int,
    *,
    constant: int | None = None,
    hamming_weight: int | None = None,
) -> int:
    h_c = _resolve_hamming_weight(
        n, constant=constant, hamming_weight=hamming_weight
    )
    return 5 * n - 5 + 2 * phi(h_c)


def modular_addition_gate_count(n: int) -> int:
    return 31 * n - 26


def modular_addition_gate_depth(n: int) -> int:
    return 19 * n + QTC.ceil_log2(n) - 11


def constant_modular_addition_gate_count(n: int) -> int:
    return 35 * n - 26


def constant_modular_addition_gate_depth(n: int) -> int:
    return 19 * n + QTC.ceil_log2(n) - 7


def unsigned_product_addition_gate_count(n: int, m: int) -> int:
    w = QTC.ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_gate_count(w, m + w)
        + (s - 1) * addition_gate_count(m + w)
        + (m + w)
    )


def unsigned_product_addition_gate_depth(n: int, m: int) -> int:
    w = QTC.ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_gate_depth(w, m + w)
        + (s - 1) * addition_gate_depth(m + w)
        + 1
    )


def positive_constant_multiplication_gate_count(
    n: int, m: int, *, constant: int | None = None
) -> int:
    ca_constant = None if constant is None else constant - 1
    return (
        unsigned_product_addition_gate_count(n - 1, m)
        + controlled_constant_addition_gate_count(
            n + m, constant=ca_constant
        )
        + 3 * n
        + m
        + 2
    )


def positive_constant_multiplication_gate_depth(
    n: int, m: int, *, constant: int | None = None
) -> int:
    ca_constant = None if constant is None else constant - 1
    return (
        unsigned_product_addition_gate_depth(n - 1, m)
        + controlled_constant_addition_gate_depth(
            n + m, constant=ca_constant
        )
        + 3 * n
        + m
        + 2
    )


def negative_constant_multiplication_gate_count(
    n: int, m: int, *, constant: int | None = None
) -> int:
    abs_constant = None if constant is None else abs(constant)
    return (
        unsigned_product_addition_gate_count(n - 1, m)
        + controlled_constant_addition_gate_count(
            n + m, constant=abs_constant
        )
        + controlled_constant_addition_gate_count(n + m, constant=1)
        + 3 * n
        + m
        + 2
    )


def negative_constant_multiplication_gate_depth(
    n: int, m: int, *, constant: int | None = None
) -> int:
    abs_constant = None if constant is None else abs(constant)
    return (
        unsigned_product_addition_gate_depth(n - 1, m)
        + controlled_constant_addition_gate_depth(
            n + m, constant=abs_constant
        )
        + controlled_constant_addition_gate_depth(n + m, constant=1)
        + 3 * n
        + m
        + 2
    )


def unsigned_modular_product_addition_gate_count(n: int, m: int) -> int:
    del m
    w = QTC.ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_gate_count(w, n)
        + (s - 1) * modular_addition_gate_count(n)
        + n
    )


def unsigned_modular_product_addition_gate_depth(n: int, m: int) -> int:
    del m
    w = QTC.ceil_log2(n)
    s = math.ceil(n / w)
    return (
        2 * s * table_lookup_gate_depth(w, n)
        + (s - 1) * modular_addition_gate_depth(n)
        + 1
    )


def constant_modular_multiplication_gate_count(n: int, m: int) -> int:
    return unsigned_modular_product_addition_gate_count(n, m)


def constant_modular_multiplication_gate_depth(n: int, m: int) -> int:
    return unsigned_modular_product_addition_gate_depth(n, m)


def constant_division_gate_count(
    n: int, m: int, *, constant: int | None = None
) -> int:
    r"""Gate-count composition of ``D_c(n,m)`` from Section 3.2.

    ``n`` is the full dividend width and ``m`` is the remainder width.  The
    two serial ranges are exactly ``i=m,...,n-1`` and ``i=m+1,...,n-1``.
    If the divisor is omitted, every internal constant-loading register is
    assigned its worst-case Hamming weight.
    """

    if n <= 0 or m <= 0 or m >= n:
        raise ValueError("constant division requires 0 < m < n")
    if constant is None:
        # GC(A_c(i)) = GC(CA_c(i)) = 9i-8 in the worst case;
        # the +2 in each bracket therefore gives 9i-6.
        first_count = n - m
        second_count = n - m - 1
        first = 9 * _sum_integers(m, n - 1) - 6 * first_count
        second = 9 * _sum_integers(m + 1, n - 1) - 6 * second_count
        return first + second + n - m - 1

    first = sum(
        controlled_constant_addition_gate_count(i, constant=constant) + 2
        for i in range(m, n)
    )
    second = sum(
        constant_addition_gate_count(i, constant=constant) + 2
        for i in range(m + 1, n)
    )
    return first + second + n - m - 1


def constant_division_gate_depth(
    n: int, m: int, *, constant: int | None = None
) -> int:
    r"""Gate-depth composition of ``D_c(n,m)`` from Section 3.3."""

    if n <= 0 or m <= 0 or m >= n:
        raise ValueError("constant division requires 0 < m < n")
    if constant is None:
        first_count = n - m
        second_count = n - m - 1
        first = (
            5 * _sum_integers(m, n - 1)
            - 3 * first_count
            + 2 * _sum_phi(m, n - 1)
        )
        second = 5 * _sum_integers(m + 1, n - 1) - second_count
    else:
        first = sum(
            controlled_constant_addition_gate_depth(i, constant=constant) + 2
            for i in range(m, n)
        )
        second = sum(
            constant_addition_gate_depth(i) + 2
            for i in range(m + 1, n)
        )
    return first + second + 1


def sum_gate_count(k: int, n: int) -> int:
    h = QTC.ceil_log2(k)
    return (
        2
        * sum(
            math.ceil(k / (2**i)) * (addition_gate_count(n + i) + 3)
            for i in range(1, h + 1)
        )
        + n
        + h
    )


def sum_gate_depth(k: int, n: int) -> int:
    h = QTC.ceil_log2(k)
    return 2 * sum(addition_gate_depth(n + i) for i in range(1, h + 1)) + 2 * h + 2


def mpmct_gate_count(controls: int) -> int:
    return 2 * controls - 3


def mpmct_gate_depth(controls: int) -> int:
    return 2 * QTC.ceil_log2(controls) - 1


# ---------------------------------------------------------------------------
# Optimized Kyber QROM pair helpers
# ---------------------------------------------------------------------------


def kyber_qrom_pair_gate_bound(eta_1: int, output_bits: int) -> int:
    """Return ``34 + 4*eta_1*output_bits`` from Eq. (Kyber QROM bound)."""

    if eta_1 not in (2, 3):
        raise ValueError("eta_1 must be 2 or 3")
    return 34 + 4 * eta_1 * output_bits


def kyber_linear_qrom_pair_gate_cost(
    eta_1: int, output_bits: int, coefficient: int
) -> int:
    """Exact GC=GD for one linear-product QROM and its inverse."""

    h_sum = sum(
        twos_complement_hamming_weight(coefficient * x, output_bits)
        for x in range(-eta_1, eta_1 + 1)
    )
    return 34 + 2 * h_sum


def kyber_modular_qrom_pair_gate_cost(
    eta_1: int, output_bits: int, coefficient: int, q: int = 3329
) -> int:
    """Exact GC=GD for one modular-product QROM and its inverse."""

    h_sum = sum(
        (coefficient * x % q).bit_count() for x in range(-eta_1, eta_1 + 1)
    )
    return 34 + 2 * h_sum


# ---------------------------------------------------------------------------
# Gate count/depth of the Kyber component circuits (Section 6)
# ---------------------------------------------------------------------------


def _metric(count: int, depth: int) -> dict[str, int]:
    return {"count": count, "cost": count, "depth": depth}


def compute_all(
    n: int,
    r: int,
    log2_L_j_list: Sequence[float],
    *,
    x_c: int | None = None,
    l_b_j_matrix: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Compute data-independent upper bounds for one Kyber ``Q`` iteration.

    Parameters
    ----------
    n, r, log2_L_j_list:
        The same inputs used by :func:`Q_Toffoli_cost.compute_all`.
    x_c:
        Exact number of constant-loading X gates in ``C(w_g||1)^T``.  If it is
        unavailable, the paper-consistent bound ``(k+1)l_q`` is used.
    l_b_j_matrix:
        Optional ``k x k`` matrix of bit lengths ``l_b[j][i]`` for the RED
        sign-extension terms.  If omitted, every entry uses the Kyber bound
        ``l_q=12``.
    """

    toffoli = QTC.compute_all(n, r, log2_L_j_list)
    p = toffoli["parameters"]
    m, k = p["m"], p["k"]
    l_q, l_g, l_cw = p["l_q"], p["l_g"], p["l_cw"]
    l_t, l_tr = p["l_t"], p["l_tr"]
    l_tilde_b, l_t_tilde_b, l_M = p["l_tilde_b"], p["l_t_tilde_b"], p["l_M"]
    eta_1 = p["eta_1"]

    if l_b_j_matrix is not None:
        if len(l_b_j_matrix) != k or any(len(row) != k for row in l_b_j_matrix):
            raise ValueError(f"l_b_j_matrix must have shape ({k},{k})")

    E_C = (r + 1) * QTC.ceil_log2(r + 1) + l_g - 1
    e_C = phi(l_g + QTC.ceil_log2(r + 1) - 1)
    if x_c is None:
        x_c_used = (k + 1) * l_q
        x_c_mode = "upper_bound"
    else:
        if x_c < 0:
            raise ValueError("x_c must be non-negative")
        x_c_used = x_c
        x_c_mode = "exact_input"

    qrom_cw = kyber_qrom_pair_gate_bound(eta_1, l_cw)
    cw_count = (
        k * r * qrom_cw
        + k * (r + 1) * addition_gate_count(l_t)
        + 2 * k * E_C
        + x_c_used
    )
    cw_depth = k * (qrom_cw + addition_gate_depth(l_t) + 2 * e_C)

    mj_count = (
        2 * k * negative_constant_multiplication_gate_count(l_t, l_tilde_b)
        + sum_gate_count(k, l_t_tilde_b)
    )
    mj_depth = (
        2 * negative_constant_multiplication_gate_depth(l_t, l_tilde_b)
        + sum_gate_depth(k, l_t_tilde_b)
    )

    per_j = []
    np_b_count = k * l_tr
    pipelined_depth_sum = 0
    for idx, toffoli_j in enumerate(toffoli["per_j"]):
        l_u_j, l_L_j = toffoli_j["l_u_j"], toffoli_j["l_L_j"]
        u_count = (
            2 * constant_division_gate_count(l_M + 1, l_L_j)
            + controlled_constant_addition_gate_count(l_u_j, constant=1)
            + controlled_constant_addition_gate_count(l_u_j, constant=-1)
            + l_u_j
            + 6
        )
        u_depth = (
            2 * constant_division_gate_depth(l_M + 1, l_L_j)
            + controlled_constant_addition_gate_depth(l_u_j, constant=1)
            + controlled_constant_addition_gate_depth(l_u_j, constant=-1)
            + 7
        )

        bit_lengths = (
            list(l_b_j_matrix[idx]) if l_b_j_matrix is not None else [l_q] * k
        )
        extension_widths = [max(l_tr - (l_u_j + b), 0) for b in bit_lengths]
        E_R_j = sum(extension_widths)
        e_R_j = max((phi(width) for width in extension_widths), default=0)
        l_b_j = max(bit_lengths)

        red_count = (
            2 * k * negative_constant_multiplication_gate_count(l_u_j, l_b_j)
            + k * addition_gate_count(l_tr)
            + 2 * E_R_j
            + 2 * l_u_j * (k - 1)
        )
        red_depth = (
            2 * negative_constant_multiplication_gate_depth(l_u_j, l_b_j)
            + addition_gate_depth(l_tr)
            + 2 * e_R_j
            + 2 * QTC.ceil_log2(k)
        )
        np_b_j_count = mj_count + 2 * u_count + red_count
        np_b_j_depth = mj_depth + 2 * u_depth + red_depth
        np_b_count += np_b_j_count
        pipelined_depth_sum += mj_depth + u_depth + red_depth
        per_j.append(
            {
                "j": idx + 1,
                "log2_L_j": toffoli_j["log2_L_j"],
                "l_u_j": l_u_j,
                "l_L_j": l_L_j,
                "E_R_j": E_R_j,
                "e_R_j": e_R_j,
                "u_j": _metric(u_count, u_depth),
                "REDj": _metric(red_count, red_depth),
                "NP_B_j": _metric(np_b_j_count, np_b_j_depth),
            }
        )

    np_b_depth = (
        phi(l_tr - l_t + 1)
        + per_j[0]["u_j"]["depth"]
        + pipelined_depth_sum
    )

    delta_eta = 1 if eta_1 == 2 else 0
    a_eta_count = constant_addition_gate_count(l_tr, constant=eta_1)
    a_eta_depth = constant_addition_gate_depth(l_tr)
    range_count = (
        k * (2 * a_eta_count + 2 * l_tr - 6 + delta_eta) + 2 * k - 3
    )
    range_depth = (
        2 * a_eta_depth
        + max(2 * QTC.ceil_log2(l_tr - 3) - 1, 2 + delta_eta)
        + 2 * QTC.ceil_log2(k)
        - 1
    )

    qrom_as = kyber_qrom_pair_gate_bound(eta_1, l_q)
    as_count = m * n * (qrom_as + modular_addition_gate_count(l_q))
    as_depth = m * (qrom_as + modular_addition_gate_depth(l_q))
    check0_count = 2 * m * l_q - 3
    check0_depth = 2 * QTC.ceil_log2(m * l_q) - 1
    lwe_count = (
        as_count
        + m * modular_addition_gate_count(l_q)
        + m * constant_modular_addition_gate_count(l_q)
        + check0_count
    )
    lwe_depth = (
        as_depth
        + modular_addition_gate_depth(l_q)
        + constant_modular_addition_gate_depth(l_q)
        + check0_depth
    )

    s_chi_count = (
        2 * cw_count
        + 2 * np_b_count
        + 2 * range_count
        + 2 * lwe_count
        + 1
    )
    s_chi_depth = (
        2 * cw_depth
        + 2 * np_b_depth
        + 2 * range_depth
        + 2 * lwe_depth
        + 1
    )

    tp_count, tp_depth = 13 * r, 11
    N = r * l_g
    s0_count = 4 * N - 3
    s0_depth = 2 * QTC.ceil_log2(N - 1) + 3
    q_count = s_chi_count + 2 * tp_count + s0_count
    q_depth = s_chi_depth + 2 * tp_depth + s0_depth

    return {
        "parameters": {
            **p,
            "E_C": E_C,
            "e_C": e_C,
            "X_C": x_c_used,
            "X_C_mode": x_c_mode,
            "gate_estimation": "data_independent_upper_bound",
        },
        "T_P": _metric(tp_count, tp_depth),
        "Cw_g": _metric(cw_count, cw_depth),
        "Mj": _metric(mj_count, mj_depth),
        "u_j": per_j[-1]["u_j"],
        "REDj": per_j[-1]["REDj"],
        "per_j": per_j,
        "NP_B": _metric(np_b_count, np_b_depth),
        "RangeCheck": _metric(range_count, range_depth),
        "As_prime": _metric(as_count, as_depth),
        "check_0": _metric(check0_count, check0_depth),
        "LWECheck": _metric(lwe_count, lwe_depth),
        "S_chi": _metric(s_chi_count, s_chi_depth),
        "S_0": _metric(s0_count, s0_depth),
        "Q": _metric(q_count, q_depth),
        "toffoli": toffoli,
    }


def compute_gate_depth_and_cost(
    n: int, r: int, log2_L_j_list: Sequence[float], **kwargs
) -> tuple[float, float]:
    """Return ``(log2 GD(Q), log2 GC(Q))`` for one ``Q`` iteration."""

    results = compute_all(n, r, log2_L_j_list, **kwargs)
    return math.log2(results["Q"]["depth"]), math.log2(results["Q"]["count"])


def compute_qsearch_resources_log2(
    n: int, r: int, log2_L_j_list: Sequence[float], **kwargs
) -> dict[str, float]:
    """Return all four expected QSearch resources on a base-2 log scale.

    The Toffoli initialization cost is zero.  Logical gate count/depth also
    include the initial ``T(P)`` and Hadamard shown in Figure 12.
    """

    results = compute_all(n, r, log2_L_j_list, **kwargs)
    log2_L = results["parameters"]["log2_L"]
    toffoli = results["toffoli"]

    toffoli_depth = log2_L + math.log2(toffoli["Q"]["depth"])
    toffoli_count = log2_L + math.log2(toffoli["Q"]["count"])
    repeated_gate_depth = log2_L + math.log2(results["Q"]["depth"])
    repeated_gate_count = log2_L + math.log2(results["Q"]["count"])
    initial_depth = math.log2(max(results["T_P"]["depth"], 1))
    initial_count = math.log2(results["T_P"]["count"] + 1)

    return {
        "toffoli_depth": toffoli_depth,
        "toffoli_count": toffoli_count,
        "gate_depth": _log2_sum(repeated_gate_depth, initial_depth),
        "gate_count": _log2_sum(repeated_gate_count, initial_count),
    }


def compute_qsearch_gate_log2(
    n: int, r: int, log2_L_j_list: Sequence[float], **kwargs
) -> tuple[float, float]:
    """Return full QSearch gate depth/count, including initialization.

    For ``L=2**(fr)`` iterations, Section 5 gives
    ``GC=GC(T(P))+1+L*GC(Q)`` and
    ``GD=max(GD(T(P)),1)+L*GD(Q)``.
    """

    resources = compute_qsearch_resources_log2(
        n, r, log2_L_j_list, **kwargs
    )
    return resources["gate_depth"], resources["gate_count"]


if __name__ == "__main__":
    # A quick smoke-test example; the exhaustive optimization remains in the
    # notebook because it supplies the actual Gram--Schmidt lengths.
    example_n, example_r = 512, 16
    example_k = QTC.compute_k(example_n, example_r)
    example = compute_all(example_n, example_r, [1.0] * example_k)
    print("log2 GD(Q) =", math.log2(example["Q"]["depth"]))
    print("log2 GC(Q) =", math.log2(example["Q"]["count"]))
