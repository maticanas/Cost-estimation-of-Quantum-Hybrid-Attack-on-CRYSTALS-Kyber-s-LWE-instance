# Cost Estimation of a Quantum Hybrid Attack on CRYSTALS-Kyber / ML-KEM LWE Instances

This repository contains the resource-estimation code accompanying the manuscript:

> **Designing Quantum Circuits for Quantum Hybrid Attacks: Cost Estimation on CRYSTALS-Kyber**  
> Hyojun Shin, Chanho Jeon, and Seokhie Hong

The code implements the circuit-level cost model used in the manuscript for the quantum hybrid attack on LWE, and instantiates it for flattened LWE instances associated with CRYSTALS-Kyber / ML-KEM. It estimates classical reduction cost, QSearch cost, qubit requirements, Toffoli resources, logical gate resources, and the additional Clifford+T-normalized quantities used for the NIST AES comparison.

The repository is intended to make the numerical results in the manuscript reproducible and to provide a modular baseline for replacing individual arithmetic, QROM, state-preparation, or search components with improved constructions.

---

## What this repository computes

For each supported instance, the main estimator searches over the guessed dimension `r` and BKZ block size `beta` and evaluates the manuscript's optimization proxy

\[
T_{\mathrm{total}}=
\frac{T_{\mathrm{red}}+T_{\mathrm{hyb}}}
     {p_{\mathrm{success}}},
\]

where:

- `T_red` is the classical lattice-reduction cost proxy,
- `T_hyb` is the Toffoli depth of one complete QSearch execution, and
- `p_success` is the modeled Babai success probability.

This is an **optimization proxy**, not a physical wall-clock time: the classical and quantum terms are reported separately in the output.

At the selected optimum, the code evaluates:

- qubit count,
- Toffoli depth (`TD`),
- Toffoli count (`TC`),
- total logical gate depth (`GD`),
- total logical gate count (`GC`),
- Clifford+T-normalized depth (`D_FT`),
- Clifford+T-normalized total gate count (`G_FT`),
- Clifford+T-normalized T-count (`T_FT`),
- rotation-synthesis precision and T-gate cost, and
- depth-constrained gate counts for the NIST AES comparison.

The current code supports:

| Instance | Flattened LWE dimension `n` | `eta_1` | Status |
|---|---:|---:|---|
| Kyber-256-k1 | 256 | 3 | Non-standard Bochum challenge instance |
| ML-KEM-512 | 512 | 3 | Standardized ML-KEM parameter set |
| ML-KEM-768 | 768 | 2 | Standardized ML-KEM parameter set |
| ML-KEM-1024 | 1024 | 2 | Standardized ML-KEM parameter set |

Kyber-256-k1 is included as a non-standard case study. It is not assigned a NIST security category by this code.

---

## Repository structure

```text
.
├── Cost esimation/
│   ├── Q_Toffoli_cost.py
│   ├── Q_gate_cost.py
│   ├── FT_gate_cost.py
│   ├── Total_cost_estimation.py
│   ├── Total_cost_estimation.ipynb
│   ├── qubit_number_estimation.ipynb
│   ├── update_ft_results.py
│   ├── test_resource_formulas.py
│   └── section7_results/
│
├── T(CBD) implemnation/
│   ├── CK_T_CBD2,3_L.ipynb
│   ├── CK_T_CBD[-2,2].ipynb
│   └── CK_T_CBD[-3,3].ipynb
│
├── requirements.txt
└── README.md
```

### Main files

**`Q_Toffoli_cost.py`**  
Implements the Toffoli-count and Toffoli-depth formulas for the arithmetic components and the complete attack circuit.

**`Q_gate_cost.py`**  
Implements the total logical gate count and total logical gate depth used in the manuscript's coarse logical-gate model.

**`FT_gate_cost.py`**  
Implements the additional Clifford+T normalization used for comparison with the NIST AES reference costs. It also computes the global rotation-synthesis precision budget and the resulting T-gate cost.

**`Total_cost_estimation.py`**  
Main optimization and resource-estimation engine. It performs the exhaustive `(r, beta)` search, writes resumable checkpoints, selects the global optimum, and evaluates the detailed circuit resources only at the selected optimum.

**`Total_cost_estimation.ipynb`**  
Recommended entry point for reproducing the main numerical results interactively.

**`qubit_number_estimation.ipynb`**  
Evaluates the data-independent qubit upper bound used in the manuscript appendix. This notebook is useful for reproducing the qubit-bound analysis, but it is not a prerequisite for running the main optimizer.

**`update_ft_results.py`**  
Optional post-processing utility. It adds the Clifford+T / rotation-synthesis / NIST comparison fields to already existing optimal-result files without rerunning the expensive `(r, beta)` search.

**`test_resource_formulas.py`**  
Regression tests for the arithmetic formulas, fixed-point bit lengths, modified-GSA determinant, Babai probability expression, QSearch resource calculations, and Clifford+T normalization.

### State-preparation notebooks

The notebooks under `T(CBD) implemnation/` construct and inspect the transformed centered-binomial distributions used by the attack and the state-preparation circuits for `eta_1 = 2` and `eta_1 = 3`.

These notebooks are auxiliary to the main estimator and require Qiskit in addition to the core Python dependencies.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/maticanas/Cost-estimation-of-Quantum-Hybrid-Attack-on-CRYSTALS-Kyber-s-LWE-instance.git
cd Cost-estimation-of-Quantum-Hybrid-Attack-on-CRYSTALS-Kyber-s-LWE-instance
```

### 2. Create a Python environment

Python 3.9 or later is required by the current source syntax.

```bash
python -m venv .venv
```

Activate it.

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the core dependencies

```bash
pip install -r requirements.txt
```

The current core requirements are:

- NumPy
- pandas
- SciPy
- pytest

To run the optional `T(CBD)` circuit notebooks, also install:

```bash
pip install jupyter qiskit qiskit-aer
```

If Jupyter is not already installed for the main notebooks, install it with:

```bash
pip install jupyter
```

---

## Quick start: reproduce the manuscript calculations

For a first-time user, the simplest workflow is to run the two notebooks below.

### Step 1 — qubit upper-bound analysis

Open:

```text
Cost esimation/qubit_number_estimation.ipynb
```

and run all cells from top to bottom.

This reproduces the data-independent qubit upper-bound calculation for

```text
n = 256, 512, 768, 1024.
```

This step is independent of the main optimization and can be skipped if you only want the Section 7 optimization results.

### Step 2 — main cost estimation

Open:

```text
Cost esimation/Total_cost_estimation.ipynb
```

and run all cells from top to bottom.

The main cell calls:

```python
results = run_cost_estimation(
    n_list=(256, 512, 768, 1024),
    output_dir="section7_results",
    workers=1,
)
```

This performs the complete search and automatically computes the coarse and Clifford+T-normalized resources at the selected optimum.

You do **not** need to run `update_ft_results.py` after a fresh run of `Total_cost_estimation.ipynb`.

---

## Command-line execution

The same main calculation can be run without Jupyter.

From the repository root:

```bash
cd "Cost esimation"
python Total_cost_estimation.py --n 256 512 768 1024 --workers 1
```

To evaluate only one instance, for example ML-KEM-512:

```bash
python Total_cost_estimation.py --n 512 --workers 1
```

To use multiple worker processes:

```bash
python Total_cost_estimation.py --n 256 512 768 1024 --workers 4
```

Parallelism changes only the runtime of the search; it does not change the candidate set or cost formulas.

The default output directory is:

```text
Cost esimation/section7_results/
```

A different directory can be selected with:

```bash
python Total_cost_estimation.py --output-dir my_results
```

---

## Resuming an interrupted search

The exhaustive search writes a checkpoint after each completed value of `r`:

```text
objective_search_results_256.csv
objective_search_results_512.csv
objective_search_results_768.csv
objective_search_results_1024.csv
```

If a run is interrupted, running the same command again with the same output directory resumes from the completed `r` values instead of restarting the entire search.

The checkpoint files contain a `model_version` field. If the resource formulas are changed, use a new output directory rather than mixing checkpoints generated by different formula versions.

---

## Output files

A complete run writes files such as:

```text
section7_results/
├── objective_search_results_256.csv
├── objective_search_results_512.csv
├── objective_search_results_768.csv
├── objective_search_results_1024.csv
│
├── best_cost_log2_results_256.csv
├── best_cost_log2_results_512.csv
├── best_cost_log2_results_768.csv
├── best_cost_log2_results_1024.csv
│
├── best_cost_log2_results_256.json
├── best_cost_log2_results_512.json
├── best_cost_log2_results_768.json
├── best_cost_log2_results_1024.json
│
├── q_circuit_upper_bounds.json
├── rotation_synthesis_budget.csv
├── ft_gate_resources.csv
└── maxdepth_gate_comparison.csv
```

### `objective_search_results_<n>.csv`

Checkpoint file containing the best objective value found for each completed `r` after the exhaustive block-size search.

### `best_cost_log2_results_<n>.csv` / `.json`

The globally selected optimum for one instance, including:

- `r` and `block_size`,
- `prob_log2`,
- `T_red_log2`,
- `T_hyb_log2`,
- `total_cost_log2`,
- qubit upper bound,
- QSearch TD/TC/GD/GC,
- resources per successful attack,
- rotation-synthesis precision,
- Clifford+T resources, and
- MAXDEPTH comparison values.

### `rotation_synthesis_budget.csv`

Summarizes the rotation-synthesis quantities used in the NIST-comparison model, including

\[
N_R = 14rL,
\]

\[
b_{\rm rot}=\left\lceil \log_2(28rL) \right\rceil,
\]

and

\[
T_R = 4b_{\rm rot}+10.
\]

### `ft_gate_resources.csv`

Contains `D_FT`, `G_FT`, and `T_FT`, both for one modeled QSearch execution and per successful key recovery.

### `maxdepth_gate_comparison.csv`

Contains the idealized depth-constrained comparison used in the manuscript for

```text
MAXDEPTH = 2^40, 2^64, 2^96.
```

For the standardized ML-KEM instances, the code uses the NIST AES reference exponents used in the current manuscript:

| ML-KEM instance | AES reference exponent |
|---|---:|
| ML-KEM-512 | 170 |
| ML-KEM-768 | 233 |
| ML-KEM-1024 | 298 |

The depth-constrained comparison is an **idealized normalization for comparison purposes**. It should not be interpreted as a claim that the required search-space partitioning can be implemented without communication, scheduling, or other architectural overhead.

---

## Optional: update existing results without rerunning the search

If `section7_results/` already contains completed

```text
best_cost_log2_results_<n>.csv
```

files from the exhaustive optimization, you can add or regenerate only the Clifford+T, rotation-synthesis, and NIST comparison results by running:

```bash
cd "Cost esimation"
python update_ft_results.py
```

This script does **not** rerun the `(r, beta)` optimization.

Use it only when you want to reuse already computed optima. It is not required when running `Total_cost_estimation.ipynb` or `Total_cost_estimation.py` from scratch.

---

## Verify the implementation

Run the regression tests from the `Cost esimation` directory:

```bash
cd "Cost esimation"
pytest -q
```

For the current repository version, the expected result is:

```text
12 passed
```

The tests check, among other things:

- the constant-division Toffoli composition,
- logical gate-count/depth compositions,
- fixed-point register-length consistency,
- the modified-GSA determinant,
- the incomplete-beta representation of the Babai probability model,
- agreement between fast and detailed Toffoli-depth calculations,
- QROM upper bounds,
- rotation-synthesis precision values, and
- Clifford+T / MAXDEPTH resource formulas.

---

## Reference results

The current implementation selects the following optima for the manuscript instances:

| Instance | `r` | `beta` | `log2(T_total)` | `log2(G_FT / p_success)` |
|---|---:|---:|---:|---:|
| Kyber-256-k1 | 37 | 135 | 97.918 | 108.689 |
| ML-KEM-512 | 96 | 408 | 181.902 | 193.092 |
| ML-KEM-768 | 185 | 691 | 248.946 | 259.826 |
| ML-KEM-1024 | 261 | 981 | 340.250 | 351.514 |

The corresponding rotation-synthesis parameters are:

| Instance | `b_rot` | T gates per synthesized rotation |
|---|---:|---:|
| Kyber-256-k1 | 59 | 246 |
| ML-KEM-512 | 137 | 558 |
| ML-KEM-768 | 218 | 882 |
| ML-KEM-1024 | 303 | 1222 |

Small differences in the last displayed decimal can occur if intermediate values are rounded before presentation. The CSV/JSON outputs retain higher numerical precision and should be treated as the reproducibility reference.

---

## Notes on the resource model

A few modeling choices are important when interpreting the output.

### Fixed-point precision

The Babai circuit uses a conservative fixed-point precision budget. The estimator computes the precision length in the log domain to avoid numerical cancellation for the extremely large expected QSearch repetition counts.

### Coarse logical-gate model

`Q_gate_cost.py` reports the manuscript's circuit-level logical `GC/GD` metrics. These count the logical operations used by the circuit descriptions and are useful for component-wise resource accounting.

### Clifford+T normalization

The NIST comparison does not use the coarse `GC/GD` values directly. `FT_gate_cost.py` additionally normalizes Toffoli gates and arbitrary state-preparation rotations into the Clifford+T-level model used for that comparison.

### Rotation synthesis

The rotation precision is allocated over the expected number of repeated state-preparation rotations in QSearch. The resulting precision is therefore tied to the expected-run model used by the resource estimator.

### Security interpretation

The outputs are **constructive upper bounds for the specific quantum hybrid attack implemented here**. They are not lower bounds on the quantum security of ML-KEM and do not rule out more efficient classical or quantum attacks.

The classical Bochum result for Kyber-256-k1 and the quantum-circuit estimates in this repository use fundamentally different cost models and should not be compared as if they were the same physical resource.

---

## Modifying or extending the estimator

The implementation is intentionally separated by resource layer:

1. modify a primitive circuit formula in `Q_Toffoli_cost.py` and/or `Q_gate_cost.py`,
2. keep the attack composition in `Total_cost_estimation.py`,
3. recompute the optimum, and
4. use `FT_gate_cost.py` for the corresponding Clifford+T normalization.

This makes it possible to study alternative adders, multipliers, QROM implementations, state-preparation circuits, or attack parameters without rewriting the full estimation pipeline.

When formulas affecting the objective search change, update `MODEL_VERSION` in `Total_cost_estimation.py` and use a fresh results directory so that old checkpoints are not accidentally reused.

---

## Citation

If you use this code in academic work, please cite the accompanying manuscript:

```bibtex
@article{ShinQuantumHybridKyber,
  author  = {Hyojun Shin and Chanho Jeon and Seokhie Hong},
  title   = {Designing Quantum Circuits for Quantum Hybrid Attacks: Cost Estimation on CRYSTALS-Kyber},
  note    = {Manuscript},
}
```

Please replace the bibliographic entry with the final journal metadata once the article is published.

---

## Contact

For questions about the manuscript or implementation, please open a GitHub issue or contact the authors through the information provided in the manuscript.
