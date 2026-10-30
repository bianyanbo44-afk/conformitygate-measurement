# ConformityGate: machine learning for sensor data quality ranking

Code and numerical evidence for *ConformityGate: Machine learning for cross-domain quality ranking of low-cost PM2.5 sensor data products* (Measurement, MEAS-D-26-15622).

The revision separates sensor-only assessment features from paired-reference
evaluation. It evaluates candidate ranking and matched deployment policies
separately. High AUROC does not establish superior all-deployment product
selection or authorize autonomous release. All revision analyses are post hoc.

## Environment

Use Python 3.12 in a clean virtual environment. Do not install these pinned
versions over an unrelated scientific environment.

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements.txt
python run_revision.py
```

The original scientific versions are NumPy 1.26.4, pandas 2.2.3, SciPy 1.16.3,
scikit-learn 1.3.2. Numerical results condition on this environment. Different
operating systems or fonts need not yield byte-identical figure files.

## Obtain the public raw data

Raw data are not redistributed. Download from the original repositories and
retain their licenses/attribution:

* QUANT: https://doi.org/10.5281/zenodo.10775692
* India workbook: https://doi.org/10.6078/D1RQ70
* U.S. EPA: https://doi.org/10.23719/1522388

Place files at these repository-relative locations:

```text
data/raw/quant/QUANT_SensorSystems_hourly.csv
data/raw/quant/QUANT_Reference_hourly.csv
data/raw/india/Campmier_Manuscript_Data.xlsx
data/raw/epa_us/Data_DevelopmentUSPAcorrection_210408/Full24hrdataset.csv
```

Raw-file SHA-256 values are in `raw_data_manifest.json`. These are the exact
releases used; newer releases may differ. The data source articles describe
upstream QC that cannot be undone from the released tables. India's workbook
contains three site aggregates derived from seven physical PA-II units.

## Pipeline

`run_revision.py` executes, in order:

1. Independent sensor/reference daily processing and sensor-defined windows.
2. Five-fold UK upstream isolation, comparisons and feature ablations; all-UK
   fits frozen for India/U.S.; strict no-EPA source refit.
3. Fully rebuilt leave-one-UK-group-out validation.
4. Cluster-bootstrap ranking and matched deployment-policy comparisons.
5. Moving-block criterion intervals, label stability, reference stress tests,
   channel-QC sensitivity, and ridge diagnostics.
6. Reference-blinding invariance, metric/budget checks, and a source-calibrated
   secondary release stress test.
7. Reproduction and diagnosis of the archived logistic score.
8. Second-round analyses: rankings by measured assessment errors (an oracle
   reference point), a label without the R2 criterion, concentration-spread
   strata, and a fully rebuilt run with a 22-hour sensor-day completeness threshold
   and common-feature checks.
9. Figure export from saved results, with a portable panel-geometry check.

The figure suite includes seven main figures and a supplementary six-panel
ROC/precision-recall diagnostic. Editable PDF/SVG and 600-dpi PNG/TIFF files
are accompanied by figure-specific source data. The within-pipeline display
reports observed-cluster bootstrap intervals and class counts; one-positive-
cluster strata use open points without an interval. These displays do not
refit models or change the archived point results.

Cache files are reused for identical inputs. Use a fresh checkout if changing
raw inputs or the analysis specification. No target label tunes a model,
feature subset, threshold, or calibration parameter. The full primary score
remains fixed even when an ablation performs better externally.

## Interpreting deployment policies

`B` is the number of prior-reference days per deployment. `q` is the fraction
of deployments reviewed. These budgets describe different resources.

The nomination-and-review policy first chooses the largest-score candidate
in each deployment and then orders those nominated products by score. Its
fixed-EPA comparator nominates the EPA product and uses unranked,
tie-averaged review. At B=0, the U.S. precision gain of 27.06 percentage
points at q=5% and q=10% belongs to this complete policy comparison.

Scoring an unchanged fixed-EPA product pool is a distinct review policy.
Its evidence is in the `epa_correction`, B=0 rows of
`revision_20261029/results/within_pipeline.csv`. In the U.S. this gives
AUROC 0.638 and precision 50.00%, 70.83%, and 59.57% at q=5%, 10%, and
20%, respectively. The nomination-and-review gain must not be assigned to
this fixed-pipeline policy. All-deployment conformity yield is evaluated
separately from review precision in `deployment_policies.csv`.

## Scope of the daily-processing sensitivity

All calibration models fit and predict daily means; aggregation precedes
calibration. The sensitivity applies a 22-of-24-hour sensor-day threshold
to UK and Indian data as an approximate match to U.S. native-record
completeness. U.S. daily values retain their released QC, and unavailable
raw subdaily records prevent exact harmonization. The common-feature run
also removes valid-hour and channel fields unavailable in the U.S.

The rebuilt common-feature AUROC is 0.962 in the UK and 0.868 in the U.S.
The Indian sensitivity retains eight evaluable blocks, AUROC 0.623
(cluster-bootstrap interval 0.412-0.860). These results concern the tested
completeness and feature settings, rather than every possible processing
difference. Full results are in `revision_20261029/results/harmonized/`.

## Evidence

`revision_20261029/results/` contains full benchmarks, fixed-budget policies,
paired contrasts, within-pipeline counts (including undefined single-class
AUROCs), exclusion distributions, per-criterion intervals, and integrity
checks. Reference perturbations are conditional scenarios, not certified
instrument uncertainty budgets. Indian block-bootstrap intervals condition
on the observed site aggregates; three-site inference is descriptive.

The directory named `data/private_labels` describes the historical separation
of feature and outcome artifacts; these are derived public sensor outcomes,
not private human-subject data. They are used only for the archived diagnostic.

`archive/submitted_specification.zip` preserves the original specification and
results; these must not be confused with revised results. The analysis lock
records revised methods before revised outcomes were computed, after the
submitted outcomes and reviews were already known.

No reviewer reports or author correspondence are included. `SHA256SUMS.json`
provides a per-file integrity manifest, excluding the manifest itself.

Large scientific artifacts use checksum-verified multipart storage. Run `python materialize_artifacts.py` to restore their normal filenames; `run_revision.py` does this automatically. `ARTIFACT_PARTS.json` specifies exact bytes and SHA-256 values. No numerical values are changed by this storage format.
