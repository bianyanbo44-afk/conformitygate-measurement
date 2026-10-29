# Reviewer-directed revision analysis specification

Locked before computing revised outcomes, 29 October 2026. Original submitted
outcomes and all reviewer feedback are known. These are reviewer-directed,
post hoc analyses, not a new prospective confirmation. Original artifacts remain
unchanged under the project root; revised artifacts are written here.

## Fixed design

1. Build sensor daily summaries without assessment-reference values or masks.
   Reference daily means are computed separately. A sensor day requires 18
   complete sensor hours; a reference day separately requires 18 reference hours.
   Select greedy non-overlapping sensor-eligible 30-day windows (>=23 sensor
   days), after a 14-day history. Evaluate labels only with >=23 paired days.
   Audit dates may use available paired dates in the preceding 14 days only.
2. Preserve the six original calibration algorithms and hyperparameters. UK
   outer evaluation uses five GroupKFold folds by site-device group. Rebuild
   every upstream learned artifact without held groups; also exclude target
   site and target device when generating each UK group's calibration outputs.
   External calibrators use UK paired days only. No external tuning.
3. Preserve the published composite label and HGB classifier hyperparameters.
   Report original-versus-revised denominators and processing effects openly.
4. Benchmark HGB against logistic, a training-only variant prior, support and
   audit heuristics, supervised predicted RMSE and absolute bias, and predicted
   normalized constraint violation. Regression targets are training labels only.
5. Feature ablations use identity (I), sensor/environment/context (E), output
   summaries (P), their seven nonempty combinations, without month, without
   valid-hour/channel fields, and without audit. All use the same outer folds.
6. Operational policies compare HGB, fixed EPA, source-selected fixed pipeline,
   variant prior, and error regressors at each fixed B=0/1/3/7 on matched blocks.
   UK fixed-pipeline selection is outer-training-only; final external choice
   uses all UK training blocks. No target label is used for selection.
   Separate candidate top-q precision from one-product-per-deployment yield.
   q=0.05/0.10/0.20/0.30. Count reference days as shared across software variants.
7. U.S. no-EPA analyses first remove EPA candidates from frozen scores, then
   remove EPA candidates AND EPA-derived model-disagreement features and refit
   UK score models. Report within-pipeline and budget-stratified rankings.
8. Cluster uncertainty: UK site-device groups, U.S. collocation identifiers;
   India report block-conditional intervals and site-cluster sensitivity (three
   site aggregates). 2000 fixed-score paired bootstrap replicates for principal
   comparisons. Report leave-one-cluster-out metric sensitivity and UK fully
   retrained leave-one-group-out evaluation.
9. Label stability: paired moving-block resampling within each 30-day window,
   circular block lengths 3 and 5 days, 500 replicates each. Individual 95%
   percentile intervals and empirical composite pass stability. These describe
   finite-sample stability, not a full metrological uncertainty budget.
   Threshold grid: slope half-width .30/.35/.40; |intercept| 4.5/5/5.5;
   R2 .68/.70/.72; RMSE 6.5/7/7.5; NRMSE 27/30/33. Report one-at-a-time
   perturbations and jointly tighter/looser definitions, without replacing
   the primary label. Only source-supported reference-uncertainty scenarios
   may be interpreted instrumentally; hypothetical scenarios are labelled.
10. QUANT A/B QC sensitivity applies daily absolute difference >5 ug/m3 AND
    relative difference >61%; report exclusion flow and consequences for EPA
    outputs. India raw channel recovery is unavailable and must be stated.
11. Diagnose logistic orientation/feature drift and ridge scaling/coefficients
    without choosing a revised model from external outcomes. Verify actual
    shrinkage coefficients .25/.50/.70 and daily calibration order.

## Deliverables

Machine-readable result tables, analysis scripts, clean/marked manuscript,
supplement with full diagnostics, one standalone response per reviewer, editor
cover letter, public reproducibility repository and archived release. Every
comment must map to evidence or a clearly explained data limitation and
alternative. No claim of completed analysis precedes its verified output.
