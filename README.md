# Wind Turbine Detection with GEE and Random Forest

This project uses Google Earth Engine (GEE), Sentinel-1/2 imagery, and random forests to identify candidate wind turbine sites at regional scale. Its workflow covers sample-coordinate processing, candidate negative-sample generation, multisource feature classification, accuracy assessment, and raster export.

## Workflow

```text
Source location data
    -> Filter records and transform coordinates
    -> Prepare positive and candidate negative samples
    -> Upload tables to GEE Assets
    -> Extract Sentinel-1/2 features and train a random forest
    -> Predict the target region and filter by probability and connected pixels
    -> Display results and export rasters
```

| Component | File | Purpose |
| --- | --- | --- |
| GEE classification and mapping | `code/wind_turbine_detection.js` | Load feature records, train and validate a classifier, predict, and export results |
| Positive-sample preparation | `code/extract_china_wind.py` | Filter records for China and transform coordinates to WGS84 |
| Negative-sample generation | `code/generate_negative_samples.py` | Generate regional candidate points outside neighborhoods of known positive samples |

## Method

The classifier is designed around 25 features from Sentinel-2 spectral bands and indices, Sentinel-1 SAR, image texture, and seasonal NDVI. The training script randomly splits a precomputed feature table and balances classes in the training subset only. It then generates a probability map, with configurable probability and minimum connected-pixel filters.

The existing sample-processing records contain 5,106 positive locations and 13,146 candidate negative locations. These are table-record counts, not counts of actual turbines, and do not imply that every record has been independently verified.

## Validation Results

The recorded confusion matrix is shown below. Rows are reference classes and columns are predictions.

| Reference / predicted class | Non-turbine | Turbine |
| --- | ---: | ---: |
| Non-turbine | 3,162 | 208 |
| Turbine | 374 | 1,315 |

| Metric | Result |
| --- | ---: |
| Validation records | 5,059 |
| Overall accuracy | 88.50% |
| Kappa | 0.7349 |
| Turbine precision | 86.34% |
| Turbine recall | 77.86% |
| Turbine F1 | 81.88% |

These metrics describe this held-out validation set. They do not establish national mapping accuracy or spatially independent validation. Spatial correlation may be present among validation records. Small-target omissions, candidate-negative label quality, and the temporal alignment between training locations and imagery require further investigation. Local map examples cannot replace an independently sampled accuracy assessment.

## Run

From the repository root, prepare the sample tables with:

```powershell
python code/extract_china_wind.py
python code/generate_negative_samples.py
```

Upload the generated tables to your own Earth Engine account. Set `positiveAsset`, `negativeAsset`, and `trainingAsset` in `code/wind_turbine_detection.js`, then load the script in the GEE Code Editor.

The classification script also requires a precomputed feature table containing the 25 configured feature fields and a `class` label. The positive and negative location CSVs do not generate this feature table. The current script defaults are for a configured example region; do not assume the recorded validation metrics were produced with exactly the same settings.

The local Python scripts require `pandas`, `numpy`, `scipy`, and `pyproj`. Project documents, screenshots, and sample data are kept locally and are not published in this repository.
