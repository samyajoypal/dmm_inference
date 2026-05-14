# Likelihood-Based Inference for Dirichlet Mixture Models

This repository contains the Python implementation and reproducible experiments for the paper:

> **Likelihood-Based Inference for Dirichlet Mixture Models via Unconstrained Parametrization**

The repository includes:

- Simulation studies for parameter estimation, consistency analysis, and hypothesis testing
- Real-data analyses using TCGA BRCA and LUAD gene expression datasets
- High-dimensional Dirichlet mixture model implementations
- Likelihood-based inference procedures including Wald, Score, and Likelihood Ratio Tests

---

# Repository Structure

```text
.
├── sim_dmm/
│   ├── run_sims.py
│   ├── run_consistency.py
│   ├── run_tests_suite_full.py
│   └── ...
│
├── real_data/
│   ├── run_brca_analysis.py
│   ├── run_luad_analysis.py
│   └── ...
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone git@github.com:samyajoypal/dmm_inference.git
cd dmm_inference
```

Install all required Python packages:

```bash
pip install -r requirements.txt
```

We recommend using a virtual environment.

Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Simulation Studies

All simulation experiments are located inside the `sim_dmm` directory.

Change into the simulation directory:

```bash
cd sim_dmm
```

## 1. Parameter Estimation Simulations

Run the main simulation study:

```bash
python run_sims.py
```

This script performs:

- Parameter estimation
- Standard error estimation
- Confidence interval evaluation
- Coverage probability analysis

---

## 2. Consistency Experiments

Run the consistency study:

```bash
python run_consistency.py
```

This script evaluates:

- RMSE behavior as sample size increases
- Empirical consistency of parameter estimators

---

## 3. Hypothesis Testing Simulations

Run the testing framework:

```bash
python run_tests_suite_full.py
python run_power_curves.py
```

This script performs:

- Wald tests
- Score tests
- Likelihood Ratio Tests (LRT)

for multiple null hypotheses under both:

- Empirical size studies
- Empirical power studies

---

# Real Data Analysis

The real-data applications are located inside the `real_data` directory.

Change into the directory:

```bash
cd real_data
```

---

## 1. Breast Cancer (TCGA BRCA)

Run the BRCA analysis:

```bash
python run_brca_analysis.py
```

This script performs:

- High-dimensional DMM clustering
- Parameter estimation
- Inference for mixing proportions and precision parameters
- Gene-level differential compositional analysis
- t-SNE visualization
- Confidence interval construction

---

## 2. Lung Cancer (TCGA LUAD)

Run the LUAD analysis:

```bash
python run_luad_analysis.py
```

This script performs:

- High-dimensional DMM clustering
- Parameter estimation
- Inference procedures
- Differential gene analysis
- Visualization and uncertainty quantification

---

# Main Features

The repository implements:

- Soft EM estimation for Dirichlet mixture models
- High-dimensional Dirichlet parameter estimation
- Louis information matrix
- Empirical score-based covariance estimation
- Wald confidence intervals
- Likelihood Ratio Tests (LRT)
- Score (Lagrange Multiplier) tests
- High-dimensional approximation methods

---

# Datasets

The real-data analyses use publicly available TCGA datasets:

- TCGA BRCA (Breast Cancer)
- TCGA LUAD (Lung Adenocarcinoma)

After preprocessing and removal of zero-valued genes, the raw read counts are converted into compositional data before fitting the Dirichlet mixture models.

---

# Reproducibility

All figures, tables, simulation studies, and real-data analyses reported in the paper can be reproduced directly from the scripts provided in this repository.

---

# Citation

If you use this repository or the associated methodology in your work, please cite the corresponding paper.

---

# Contact

For questions, suggestions, or collaborations, please contact:

**Samyajoy Pal**  
RPTU Kaiserslautern-Landau  
Chair of Data Analytics  
Email: samyajoy.pal@rptu.de
