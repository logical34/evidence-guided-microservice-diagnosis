# Evidence-Guided Microservice Diagnosis

This repository contains the final incident-level records and analysis code for the manuscript **Evidence-Guided Fault-Type Revision and Hierarchical Release for Microservice Diagnosis**.

The method keeps the service nominated by a language model, revises the fault type from numerical evidence, and then releases a complete diagnosis, a service-only conclusion, or a referral. The saved records cover the RCAEval evaluation, four provider response streams, and the separate Eadro test.

## Reproduce the analysis

Install Python 3.13 or a compatible recent version, then run:

```bash
python -m pip install -r requirements.txt
python scripts/verify_results.py
python scripts/make_tables.py
python scripts/make_figures.py
```

The first command checks the reported counts, data partitions, release thresholds, provider summaries, and external test totals. The other two commands rebuild the LaTeX tables in `tables/` and the quantitative figures in `figures/` as PDF, SVG, and PNG files. No API key or LLM request is used.

## Main result

The primary RCAEval evaluation contains 645 incidents with root-service groups isolated between fitting, calibration, and testing.

| Output | Count |
|---|---:|
| Complete diagnoses | 397 |
| Service-only conclusions | 162 |
| Referrals | 86 |
| Errors among complete diagnoses | 34 |
| Errors among all released conclusions | 40 |

Complete coverage is 61.55%, error among complete diagnoses is 8.56%, and error among all released conclusions is 7.16%. The four difficult CPU, disk, delay, and loss types have a complete-diagnosis error of 12.88%. The separate 45-incident Eadro test has 23 complete releases and 3 errors.

These are retrospective empirical results. They are not population-level risk guarantees.

## Repository layout

```text
data/
  retrospective/       Fold definitions and fitting/calibration/test partitions
  revision/            RCAEval actions, selected rules, subgroup and provider records
  external/            Eadro actions, controls, cohort definition, and request costs
scripts/
  method.py            Score normalization, service prior, and release rule
  verify_results.py    Independent count, partition, and threshold checks
  make_tables.py       Manuscript and supplementary tables
  make_figures.py      Three quantitative figures and their source CSV files
figures/               Saved PDF versions of the quantitative figures
tables/                Generated LaTeX tables
```

In the action files, `fixed_service` is the proposed anchored method. The `llm` and `llm_direct` rows contain the initial language-model outputs. Other methods are controls. `budget_summary.csv` contains the threshold-sweep summaries used by the risk-coverage figure.

## Scope

This is a compact analysis repository. It starts from saved incident-level outputs and does not retrain the numerical models or repeat the LLM requests. Raw telemetry, feature caches, model checkpoints, old analysis versions, and manuscript sources are not included.

The original datasets are available from their authors:

- RCAEval: <https://doi.org/10.1145/3701716.3715290>
- Eadro data: <https://doi.org/10.5281/zenodo.7615394>

The use and redistribution of source data and derived artifacts remain subject to the original licenses and terms; see [Data Use and Licensing](DATA_USE.md).

The Eadro results use the original development/calibration/test allocation and are reported separately from the RCAEval cross-validation results.
