# Wind Turbine Detection

Google Earth Engine and random-forest experiments for identifying wind turbines from multisource satellite imagery.

## Repository layout

- `code/`: Earth Engine workflow and local sample-processing scripts.
- `data/`: local-only input and sample CSV files; excluded from Git.
- `docs/`: local project and submission documents; excluded from Git.
- `reference/`: local reference material; excluded from Git.

## Workflow

Run from the repository root:

```powershell
python code/extract_china_wind.py
python code/generate_negative_samples.py
```

Upload the resulting sample tables to your own Earth Engine account. Set `positiveAsset`, `negativeAsset`, and `trainingAsset` in `code/wind_turbine_detection.js` to assets you own before running it in the Earth Engine Code Editor.

The Earth Engine script also requires a precomputed training-feature table with the configured feature fields. The local sample CSV files do not recreate that table.

Python dependencies: `pandas`, `numpy`, `scipy`, and `pyproj`.
