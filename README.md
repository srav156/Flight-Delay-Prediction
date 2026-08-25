# Flight Delay Prediction — Two-Stage Machine Learning Engine

**Author:** Sibi Ravi

A machine learning project that predicts flight arrival delays in two stages: first classifying whether a flight will be delayed by 15 minutes or more, then, for flights predicted as delayed, regressing how many minutes late it will arrive. Built on 1,851,436 US domestic flights (2016–2017) across 15 major airports, merged with hourly weather observations at each airport.

A full write-up with methodology, figures, and results is in [`Flight_Delay_Prediction_SibiRavi.pdf`](Flight_Delay_Prediction_SibiRavi.pdf) — this README summarizes it and maps it to the code.

## Problem

A flight is defined as delayed if it arrives 15+ minutes after its scheduled arrival time (`ArrDel15`, a binary flag). Given that a flight is delayed, `ArrDelayMinutes` gives the actual size of the delay in minutes. The engine predicts both:

1. **Classification** — will this flight be delayed? (Yes/No)
2. **Regression** — if delayed, by how many minutes?

## Data

- **Flight data:** every US domestic flight in 2016–2017, filtered down to 15 airports (ATL, CLT, DEN, DFW, EWR, IAH, JFK, LAS, LAX, MCO, MIA, ORD, PHX, SEA, SFO) and 17 relevant columns; rows with nulls in selected columns were dropped.
- **Weather data:** hourly weather readings at those same 15 airports (wind speed/direction, weather code, precipitation, visibility, pressure, cloud cover, dew point, wind chill, temperature, humidity).
- **Merge:** flight and weather records were joined on date, hour (flight times rounded to the nearest hour), and airport, producing **1,851,436** merged flight records.
- **Feature selection:** a correlation heatmap was used to drop redundant/near-duplicate features (r ≈ 1.0, or r > 0.90), and any column that would leak arrival information was removed. The final feature set: `WindSpeedKmph`, `WindDirDegree`, `WeatherCode`, `precipMM`, `Visibility`, `Pressure`, `Cloudcover`, `DewPointF`, `tempF`, `Humidity`, `Year`, `Month`, `DepDelayMinutes`, `Origin`, `Dest` — plus the two targets, `ArrDel15` (classification) and `ArrDelayMinutes` (regression).
- **Class imbalance:** only ~21% of flights are delayed. A single stratified 80/20 train/test split was done first, and SMOTE oversampling was applied to the training set only, so the test set stays composed of real, untouched observations.

## Notebooks

| Notebook | Purpose |
|---|---|
| `FlightDataPreprocessing.ipynb` | Cleans and filters the raw flight dataset (airport/column filtering, null removal, monthly merge). |
| `WeatherDataPreprocessing.ipynb` | Cleans and filters the raw weather dataset down to the 15 selected airports and variables. |
| `FlightWeatherMerge.ipynb` | Merges the cleaned flight and weather data on date/hour/airport into the combined dataset. |
| `ClassificationModels3.ipynb` / `Classification3.ipynb` | Stage 1 exploration: trains and compares Logistic Regression, Decision Tree, XGBoost, Random Forest, and Extra Trees classifiers on `ArrDel15`; produces confusion matrices and metrics (precision, recall, F1, accuracy, AUC). |
| `Regression3.ipynb` | Stage 2 exploration: trains and compares Linear Regression, Decision Tree, XGBoost, Random Forest, and Extra Trees regressors on `ArrDelayMinutes` (RMSE, MAE, R²), evaluated across the whole dataset (not just delayed flights). |
| `Pipeline2.ipynb` | Combines the best classifier and regressor into the final two-stage pipeline: classify delay → for flights predicted delayed, regress the delay minutes. Includes the bin-by-bin performance breakdown (15–200, 200–400, 400–600, 600–800, 800–1000, 1000+ minutes). |

## Data folder

`Data/` holds the intermediate and final CSVs produced by the notebooks above (raw merged data, train/test feature and label splits, SMOTE-resampled sets, regression-specific sets) along with the trained model files:

- `LogisticRegression.joblib`, `DecisionTreeClassifier.joblib`, `GradientBoostingClassifier.joblib`, `XGBClassifier.joblib` — Stage 1 classifiers evaluated during model comparison.
- Subfolders `Classification3/`, `Regression3/`, `Pipeline2/`, `FlightData/`, `Flight Info/`, `weather/` hold per-notebook working data.

Note: these CSVs are large (some 100MB+) and are working data, not meant to be committed to source control as-is.

## Results

**Stage 1 — Classification** (best model: **Random Forest**)

| Model | Precision (delayed) | Recall (delayed) | F1 (delayed) | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.78 | 0.76 | 0.90 | 0.903 |
| Decision Tree | 0.66 | 0.70 | 0.68 | 0.86 | 0.805 |
| XGBoost | 0.81 | 0.74 | 0.77 | 0.91 | 0.915 |
| **Random Forest** | 0.77 | 0.76 | 0.77 | 0.90 | **0.919** |
| Extra Trees | 0.78 | 0.73 | 0.76 | 0.90 | 0.908 |

**Stage 2 — Regression** (best model: **Extra Trees**, on all flights)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 11.110 | 6.812 | 0.928 |
| Decision Tree | 15.149 | 7.453 | 0.865 |
| XGBoost | 13.007 | 6.796 | 0.901 |
| Random Forest | 10.728 | 6.085 | 0.932 |
| **Extra Trees** | 10.767 | 6.043 | **0.932** |

**Full pipeline** (Random Forest classifier → Extra Trees regressor, on flights predicted delayed only):

- Classification: F1 = 0.77 (delayed class), Accuracy = 0.90, AUC = 0.920
- Regression: RMSE = 19.898 min, MAE = 15.208 min, R² = 0.929
- 94.66% of delayed flights fall in the 15–200 minute range, where the model performs best (RMSE 17.10, MAE 12.38) — the higher aggregate RMSE is driven by a small number of extreme-delay flights (200+ minutes) that square-error weights heavily.

## Known limitation

`DepDelayMinutes` (departure delay) is used as a feature. This carries some leakage risk — a late departure strongly predicts a late arrival — but it's kept because it also captures non-weather delay causes (mechanical issues, runway congestion) that the weather features alone can't. This is called out explicitly in the report as an area for future improvement.
