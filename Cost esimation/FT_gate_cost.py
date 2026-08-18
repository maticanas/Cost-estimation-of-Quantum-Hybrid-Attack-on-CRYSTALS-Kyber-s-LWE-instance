"""Clifford+T normalization used for the NIST AES comparison.

This module implements the additional resource model in Section 7.4 of the
revised manuscript ``sn-article 0818 1639.tex``.  The existing modules
``Q_Toffoli_cost`` and ``Q_gate_cost`` remain the authoritative sources for
the coarse Toffoli and logical-gate resources.  Here those four base-2
logarithmic resources are normalized to the Clifford+T logical-gate model
used in the AES resource analysis underlying NIST's current reference values.

The paper uses the following conventions.

* Every Toffoli is charged as 7 T/T^dagger gates and 8 Clifford operations,
  i.e. 15 elementary Clifford+T operations in total.  Relative to the coarse
  gate model, which already counted one Toffoli as one operation, this adds
  14 operations per Toffoli.
* One Q iteration contains 14*r arbitrary R_y rotations.  For the expected
  L applications of Q, N_R = 14*r*L.
* A global expected-run synthesis-error budget of 1/2 gives
  b_rot = ceil(log2(28*r*L)).
* The explicit synthesis charge is T_R = 4*b_rot + 10 and
  C_R = 2*T_R + 5 elementary single-qubit Clifford+T operations/depth per
  synthesized rotation.
* D_FT is the conservative layer-replacement bound
  D_G + 7*T_D + (C_R-1)*(10*L).

All expensive quantities are evaluated in the log2 domain; L itself is never
constructed.  By default the rotation budget uses the exact CBD-derived
coefficient for log2 L rather than the rounded 1.108/1.302 coefficients used
in the older coarse estimator.  This reproduces the precision entries in the
revised manuscript (e.g. 137, 218, and 303 bits at the three standardized
optima) without changing the previously optimized attack points.
"""

from __future__ import annotations

import math
from math import comb
from typing import Mapping

import Q_Toffoli_cost as QTC


MAXDEPTH_EXPONENTS: tuple[int, ...] = (40, 64, 96)

# Current NIST security-evaluation reference exponents used in the revised
# manuscript.  The depth-constrained AES reference is 2**exponent/MAXDEPTH.
NIST_AES_EXPONENTS: dict[int, int] = {
    512: 170,
    768: 233,
    1024: 298,
}


def log2_sum_many(*values: float) -> float:
    """Return log2(sum(2**x for x in values)) stably."""

    if not values:
        raise ValueError("at least one value is required")
    high = max(values)
    if math.isinf(high) and high < 0:
        return high
    return high + math.log2(sum(math.exp2(value - high) for value in values))


def cbd_probabilities(eta: int) -> list[float]:
    """Return the exact centered-binomial probabilities for CBD_eta."""

    if eta <= 0:
        raise ValueError("eta must be positive")
    denominator = 2 ** (2 * eta)
    return [
        comb(2 * eta, eta + x) / denominator
        for x in range(-eta, eta + 1)
    ]


def qsearch_exponent_from_eta(eta: int) -> float:
    r"""Return the exact coefficient f in ``log2(L) = f*r``.

    The input to QSearch is the transformed distribution T(CBD_eta).  If p_x
    are the original CBD probabilities, first set
    ``d_x = p_x**(2/3) / sum_y p_y**(2/3)``.  The one-coordinate QSearch
    factor is then

        L_1 = (sum_x d_x**(2/3))**(3/2),

    hence ``f = log2(L_1)``.  Numerically this is approximately 1.108108 for
    eta=2 and 1.302185 for eta=3.
    """

    probabilities = cbd_probabilities(eta)
    weights = [p ** (2.0 / 3.0) for p in probabilities]
    normalization = sum(weights)
    transformed = [weight / normalization for weight in weights]
    return 1.5 * math.log2(
        sum(d ** (2.0 / 3.0) for d in transformed)
    )


def qsearch_log2_L(n: int, r: int, *, exact: bool = True) -> float:
    """Return ``log2 L`` for the selected Kyber/ML-KEM instance."""

    if r <= 0:
        raise ValueError("r must be positive")
    rounded_f, eta_1, _ = QTC.kyber_instance_parameters(n)
    f = qsearch_exponent_from_eta(eta_1) if exact else rounded_f
    return f * r


def rotation_synthesis_resources(
    n: int,
    r: int,
    *,
    exact_qsearch_exponent: bool = True,
) -> dict[str, float | int]:
    """Return the rotation-synthesis budget used in Section 7.4.

    The primary ``rotation_count_log2`` follows Eq. N_R=14*r*L and therefore
    excludes the one additional initial T(P), exactly as the displayed paper
    equations do.  ``rotation_count_with_initial_log2`` is also returned for
    users who want the strict count 7*r*(2*L+1); at the reported optima it does
    not change ``b_rot``.
    """

    log2_L = qsearch_log2_L(n, r, exact=exact_qsearch_exponent)
    log2_N_R = math.log2(14 * r) + log2_L

    # epsilon_rot <= 1/(28*r*L), so b_rot = ceil(log2(28*r*L)).
    b_rot = math.ceil(math.log2(28 * r) + log2_L)
    T_R = 4 * b_rot + 10
    C_R = 2 * T_R + 5

    # 7*r*(2L+1), evaluated without constructing L.
    log2_two_L_plus_one = log2_sum_many(1.0 + log2_L, 0.0)
    log2_N_R_with_initial = math.log2(7 * r) + log2_two_L_plus_one

    return {
        "log2_L_rotation_budget": log2_L,
        "rotation_count_log2": log2_N_R,
        "rotation_count_with_initial_log2": log2_N_R_with_initial,
        "rotation_precision_bits": b_rot,
        "rotation_T_per_gate": T_R,
        "rotation_CliffordT_ops_per_gate": C_R,
        "rotation_T_count_log2": log2_N_R + math.log2(T_R),
    }


def compute_ft_resources_from_log2(
    n: int,
    r: int,
    coarse_resources: Mapping[str, float],
    *,
    exact_qsearch_exponent: bool = True,
    replace_initial_rotations: bool = False,
) -> dict[str, float | int]:
    r"""Normalize one complete expected QSearch run to Clifford+T resources.

    ``coarse_resources`` must provide the four keys returned by
    ``Q_gate_cost.compute_qsearch_resources_log2``:

    * ``toffoli_depth`` = log2(T_D)
    * ``toffoli_count`` = log2(T_C)
    * ``gate_depth`` = log2(D_G)
    * ``gate_count`` = log2(G_G)

    By default this implements the manuscript equations literally.  Set
    ``replace_initial_rotations=True`` to additionally replace the seven-r
    rotations in the one initial T(P); this is a slightly stricter full-QSearch
    normalization and is not used for the manuscript tables.
    """

    required = {"toffoli_depth", "toffoli_count", "gate_depth", "gate_count"}
    missing = required.difference(coarse_resources)
    if missing:
        raise KeyError(f"coarse_resources is missing {sorted(missing)}")

    rotation = rotation_synthesis_resources(
        n, r, exact_qsearch_exponent=exact_qsearch_exponent
    )
    log2_N_R = float(rotation["rotation_count_log2"])
    if replace_initial_rotations:
        log2_N_R = float(rotation["rotation_count_with_initial_log2"])

    b_rot = int(rotation["rotation_precision_bits"])
    T_R = int(rotation["rotation_T_per_gate"])
    C_R = int(rotation["rotation_CliffordT_ops_per_gate"])

    log2_TD = float(coarse_resources["toffoli_depth"])
    log2_TC = float(coarse_resources["toffoli_count"])
    log2_DG = float(coarse_resources["gate_depth"])
    log2_GG = float(coarse_resources["gate_count"])

    # Eq. (FT gate count): G_FT = G_G + 14*T_C + N_R*(C_R-1).
    log2_G_FT = log2_sum_many(
        log2_GG,
        math.log2(14.0) + log2_TC,
        log2_N_R + math.log2(C_R - 1),
    )

    # Eq. (FT T count): T_FT = 7*T_C + N_R*T_R.
    log2_T_FT = log2_sum_many(
        math.log2(7.0) + log2_TC,
        log2_N_R + math.log2(T_R),
    )

    # Eq. (FT depth): D_FT <= D_G + 7*T_D + (C_R-1)*D_R,
    # with D_R = 10*L for the rotation-only depth of the expected L Q runs.
    log2_L = float(rotation["log2_L_rotation_budget"])
    log2_rotation_depth = math.log2(C_R - 1) + math.log2(10.0) + log2_L
    log2_D_FT = log2_sum_many(
        log2_DG,
        math.log2(7.0) + log2_TD,
        log2_rotation_depth,
    )

    return {
        **rotation,
        "QSearch_FT_depth_log2": log2_D_FT,
        "QSearch_FT_gate_count_log2": log2_G_FT,
        "QSearch_FT_T_count_log2": log2_T_FT,
        "FT_rotation_replacement_includes_initial_TP": int(
            replace_initial_rotations
        ),
    }


def add_success_probability(
    ft_resources: Mapping[str, float | int],
    prob_log2: float,
) -> dict[str, float]:
    """Return FT depth/gate/T resources per successful key recovery."""

    return {
        "Hybrid_FT_depth_per_success_log2": float(
            ft_resources["QSearch_FT_depth_log2"]
        )
        - prob_log2,
        "Hybrid_FT_gate_count_per_success_log2": float(
            ft_resources["QSearch_FT_gate_count_log2"]
        )
        - prob_log2,
        "Hybrid_FT_T_count_per_success_log2": float(
            ft_resources["QSearch_FT_T_count_log2"]
        )
        - prob_log2,
    }


def nist_aes_reference_log2(n: int, h: int) -> int | None:
    """Return the current NIST AES reference exponent minus MAXDEPTH exponent."""

    exponent = NIST_AES_EXPONENTS.get(n)
    return None if exponent is None else exponent - h


def maxdepth_gate_count_log2(
    ft_gate_count_per_success_log2: float,
    ft_depth_log2: float,
    h: int,
) -> float:
    r"""Return log2 of the depth-constrained aggregate FT gate count.

    G_MD^FT(h) = (G_FT/p_success) * max(1, D_FT/2**h).
    """

    return ft_gate_count_per_success_log2 + max(0.0, ft_depth_log2 - h)


def maxdepth_comparison(
    n: int,
    ft_resources: Mapping[str, float | int],
    prob_log2: float,
    *,
    maxdepth_exponents: tuple[int, ...] = MAXDEPTH_EXPONENTS,
) -> list[dict[str, float | int | None]]:
    """Return the rows used for the manuscript's MAXDEPTH comparison table."""

    gate_per_success = float(ft_resources["QSearch_FT_gate_count_log2"]) - prob_log2
    depth_log2 = float(ft_resources["QSearch_FT_depth_log2"])
    return [
        {
            "n": n,
            "h": h,
            "FT_MAXDEPTH_gate_count_log2": maxdepth_gate_count_log2(
                gate_per_success, depth_log2, h
            ),
            "NIST_AES_reference_log2": nist_aes_reference_log2(n, h),
        }
        for h in maxdepth_exponents
    ]


def instance_label(n: int) -> str:
    """Human-readable name used by the generated Section 7 CSV tables."""

    if n == 256:
        return "Kyber-256-k1"
    if n in NIST_AES_EXPONENTS:
        return f"ML-KEM-{n}"
    return f"n={n}"
