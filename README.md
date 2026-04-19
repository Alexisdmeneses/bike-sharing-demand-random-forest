# Bike sharing demand — Random Forest

End-to-end machine learning pipeline to forecast hourly bike rental demand using the Kaggle Bike Sharing Demand dataset. Includes EDA, datetime feature engineering, baseline comparison, and Random Forest regressor optimized for RMSLE.

## Research question

*Which factors drive hourly bike rental demand, and how much can a Random Forest improve over a linear baseline?*

## Approach

1. **EDA** — distribution analysis, target skewness, demand patterns by hour, weekday, weather and season
2. **Feature engineering** — extraction of `hour`, `dayofweek`, `month`, `year` from datetime
3. **Target transformation** — `log1p(count)` to stabilize variance and align with RMSLE metric
4. **Baseline** — Linear Regression (RMSLE CV = 1.024)
5. **Main model** — Random Forest Regressor with 5-fold cross-validation (RMSLE CV = 0.449)
6. **Submission** — predictions reversed with `expm1`, validated against test set distribution

## Key results

- Random Forest achieved RMSLE of 0.449 — 56% improvement over linear baseline (1.024)
- `hour` identified as the most important feature: clear bimodal pattern at 8h and 17–18h (commute peaks)
- Weather condition shows strong monotonic effect: clear sky >> light rain >> storm
- Zero negative predictions in submission — all 6,493 test rows validated

## Data

- Source: [Kaggle — Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand)
- Train: 10,886 rows · 12 columns · first 19 days of each month (2011–2012)
- Test: 6,493 rows · 9 columns · days 20–end of each month

## Stack

Python · scikit-learn · pandas · numpy · matplotlib · seaborn

## Structure

```
├── Forum01_Alexis_Meneses.ipynb    ← main notebook
├── Forum01_Alexis_Meneses.pdf      ← rendered output
└── README.md
```

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
jupyter notebook Forum01_Alexis_Meneses.ipynb
```

Data must be downloaded from Kaggle:
[kaggle.com/c/bike-sharing-demand/data](https://www.kaggle.com/c/bike-sharing-demand/data)
