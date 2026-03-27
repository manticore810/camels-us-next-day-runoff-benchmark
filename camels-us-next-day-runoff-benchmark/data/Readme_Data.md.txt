# Data notes

This repository does not redistribute the full raw CAMELS-US archive.

Instead, it provides:
- the 50 benchmark gauge IDs used in the manuscript
- derived benchmark outputs
- scripts used to build the modelling-ready panel
- manuscript figures and result tables

## Primary data sources

The benchmark is based on public CAMELS-US resources:
- CAMELS-US catchment attributes
- CAMELS-US USGS streamflow files
- Daymet basin-mean forcing distributed with CAMELS-US

## Benchmark subset

The executed benchmark uses a 50-basin subset of CAMELS-US selected for a chronology-respecting next-day runoff benchmark.

The gauge IDs are listed in:
- `gauge_ids_50.txt`

## Important note

Users should obtain the original raw CAMELS-US files from the official source and then run the preprocessing scripts in this repository to reproduce the modelling-ready dataset.