"""Regression checks for the formulas used in the revised manuscript."""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.integrate import quad
from scipy.special import beta, betainc

import Q_Toffoli_cost as QTC
import Q_gate_cost as QGC
import Total_cost_estimation as TOTAL


def test_constant_division_toffoli_composition() -> None:
    for n in range(3, 50):
        for m in range(1, n):
            expanded = sum(
                QTC.controlled_constant_addition_cost(i) + 1
                for i in range(m, n)
            ) + sum(
                QTC.constant_addition_cost(i) + 1
                for i in range(m + 1, n)
            )
            expected = 2 * n**2 - 2 * m**2 - 6 * n + 4 * m + 3
            assert expanded == expected == QTC.constant_division_cost(n, m)


def test_constant_division_gate_compositions() -> None:
    for n in range(3, 35):
        for m in range(1, n):
            gc = sum(
                QGC.controlled_constant_addition_gate_count(i) + 2
                for i in range(m, n)
            ) + sum(
                QGC.constant_addition_gate_count(i) + 2
                for i in range(m + 1, n)
            ) + (n - m - 1)
            gd = sum(
                QGC.controlled_constant_addition_gate_depth(i) + 2
                for i in range(m, n)
            ) + sum(
                QGC.constant_addition_gate_depth(i) + 2
                for i in range(m + 1, n)
            ) + 1
            assert QGC.constant_division_gate_count(n, m) == gc
            assert QGC.constant_division_gate_depth(n, m) == gd


def test_fixed_point_length_partition() -> None:
    for n in QTC.SUPPORTED_DIMENSIONS:
        for r in (1, max(2, n // 5), n - 1):
            k = QTC.compute_k(n, r)
            l_r = QTC.compute_l_r(r)
            l_k = QTC.compute_l_k(k)
            l_t = QTC.compute_l_t(l_r)
            l_p = QTC.compute_l_p(
                l_t, QTC.kyber_instance_parameters(n)[0], r, k
            )
            l_M = QTC.compute_l_M(
                QTC.compute_l_t_tilde_b(
                    l_t, QTC.compute_l_tilde_b(l_p)
                ),
                l_k,
            )
            for log2_L_j in (-2.3, 0.0, 1.0, 11.999, 23.4):
                l_L_j = QTC.compute_l_L_j(l_p, log2_L_j)
                l_u_j = QTC.compute_l_u_j(
                    l_t, QTC.KYBER_QUBIT_LENGTH, l_k, log2_L_j
                )
                assert l_L_j + l_u_j == l_M
                assert (l_M + 1) - l_L_j == l_u_j + 1


def test_modified_gsa_determinant() -> None:
    for n, r, block_size in (
        (256, 37, 135),
        (512, 96, 408),
        (768, 185, 691),
        (1024, 261, 981),
    ):
        k = QTC.compute_k(n, r)
        delta = TOTAL.calculate_delta(block_size)
        profile, _ = TOTAL.modified_gsa_profile_log2(
            k, n - r, delta, verify_determinant=True
        )
        assert math.isclose(
            float(profile.sum()), n * math.log2(3329), abs_tol=1e-8
        )


def test_beta_form_against_integral() -> None:
    for dimension, ratio in ((8, 0.2), (32, 0.1), (128, 0.05)):
        integral, _ = quad(
            lambda t: (1.0 - t * t) ** ((dimension - 3.0) / 2.0),
            -1.0,
            -ratio,
        )
        integral_form = 1.0 - 2.0 * integral / beta(
            (dimension - 1.0) / 2.0, 0.5
        )
        beta_form = betainc(0.5, (dimension - 1.0) / 2.0, ratio**2)
        assert math.isclose(
            integral_form, beta_form, rel_tol=1e-10, abs_tol=1e-12
        )


def test_fast_and_detailed_toffoli_depths() -> None:
    generator = random.Random(20260815)
    for n in QTC.SUPPORTED_DIMENSIONS:
        r = max(2, n // 7)
        k = QTC.compute_k(n, r)
        profile = [generator.uniform(-3.0, 24.0) for _ in range(k)]
        detailed = QTC.compute_all(n, r, profile)["Q"]["depth"]
        assert QTC.compute_q_toffoli_depth_fast(n, r, profile) == detailed


def test_uniform_grouped_resources() -> None:
    for n in QTC.SUPPORTED_DIMENSIONS:
        r = max(2, n // 6)
        k = QTC.compute_k(n, r)
        log2_L_j = 3.25
        logs = [log2_L_j] * k
        toffoli = QTC.compute_all(n, r, logs)
        gates = QGC.compute_all(n, r, logs)
        grouped_toffoli = QTC.compute_uniform_toffoli_resources(
            n, r, log2_L_j
        )
        grouped_all = TOTAL.uniform_q_resources(n, r, log2_L_j)
        assert grouped_toffoli["Q"] == toffoli["Q"]
        assert grouped_all["toffoli_depth"] == toffoli["Q"]["depth"]
        assert grouped_all["toffoli_count"] == toffoli["Q"]["count"]
        assert grouped_all["gate_depth"] == gates["Q"]["depth"]
        assert grouped_all["gate_count"] == gates["Q"]["count"]


def test_cbd_moments_and_qrom_bounds() -> None:
    assert TOTAL.cbd_second_moment(2) == 1.0
    assert TOTAL.cbd_second_moment(3) == 1.5
    for eta in (2, 3):
        for width in (12, 14, 20):
            bound = QGC.kyber_qrom_pair_gate_bound(eta, width)
            for coefficient in (-3328, -1, 0, 1, 3328):
                exact = QGC.kyber_linear_qrom_pair_gate_cost(
                    eta, width, coefficient
                )
                assert exact <= bound


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"passed: {test.__name__}")
