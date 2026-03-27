# CAMELS-US next-day runoff benchmark

This repository contains the code, derived benchmark files, result tables, figures, and manuscript source for a chronology-respecting benchmark of next-day runoff prediction across a 50-basin CAMELS-US subset.

## Study summary

The benchmark compares four models for one-day-ahead runoff prediction:

- Persistence
- Ridge regression
- Random Forest
- HistGradientBoosting

The study uses open data only:
- CAMELS-US basin attributes and discharge
- Daymet basin-mean forcing

The benchmark is framed as a retrospective hindcast benchmark, not a latency-constrained operational forecast system.

## Repository structure

- `data/` — benchmark subset metadata and gauge IDs
- `scripts/` — preprocessing, training, and figure-generation scripts
- `results/` — derived result tables
- `figures/` — manuscript figures
- `manuscript/` — LaTeX submission files

## Main reported benchmark result

On the pooled 2005–2014 test period, Random Forest achieved the strongest overall performance:

- NSE = 0.670
- KGE = 0.756
- RMSE = 2.177 mm d^-1
- MAE = 0.569 mm d^-1

Basin-wise summaries also ranked Random Forest first, with a median basin-wise NSE of 0.623 and 27 basin wins out of 50.

## Reproducibility

The executed workflow used Python with:
- pandas
- numpy
- scikit-learn
- matplotlib

A fixed chronological split was used:
- train: 1990–1999
- validation era: 2000–2004
- test: 2005–2014

## Citation

Please cite the archived Zenodo release for the exact version used in the manuscript.

## Authors

- Zeeshan Asghar
- Muhammad Waseem

Ghulam Ishaq Khan Institute of Engineering Sciences and Technology, Topi, Khyber Pakhtunkhwa, Pakistan
