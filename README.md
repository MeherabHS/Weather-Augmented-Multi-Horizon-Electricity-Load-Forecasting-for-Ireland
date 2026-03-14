# Weather-Augmented Multi-Horizon Electricity Load Forecasting for Ireland

**Author:** Meherab Hossain Shafin  
**Department:** Software Engineering (SWE)  
**Institution:** Daffodil International University  

## Overview

This project develops a **multi-horizon electricity load forecasting pipeline for Ireland** using statistical, machine learning, and deep probabilistic models. The system integrates **ENTSO-E hourly electricity demand data** with **NASA POWER weather data** and evaluates forecasting performance across three operational horizons:

- **t+1**: 1 hour ahead  
- **t+24**: 24 hours ahead  
- **t+168**: 168 hours (7 days) ahead  

The study compares:

- Seasonal Naive baseline  
- SARIMAX  
- Quantile Gradient Boosting Regressor (GBR)  
- Weather-augmented Quantile GBR  
- DeepAR  

The final result shows that **weather-augmented Gradient Boosting** is the strongest forecasting approach for this dataset.

---

## Problem Statement

Electricity demand forecasting is critical for:

- grid stability
- generation scheduling
- operational planning
- market efficiency

Forecasting errors can increase reserve requirements, reduce dispatch efficiency, and weaken system planning. This project addresses that problem using a reproducible applied machine learning workflow.

---

## Data Sources

### 1. Electricity Load Data
Source: **ENTSO-E Transparency Platform**

Used for:
- hourly Irish electricity demand

### 2. Weather Data
Source: **NASA POWER**

Weather variables used:
- `T2M` — temperature at 2 meters
- `WS10M` — wind speed at 10 meters
- `ALLSKY_SFC_SW_DWN` — surface solar radiation

---

## Why the Project Shifted to ENTSO-E

The original data collection plan considered alternative electricity APIs. However, those sources introduced practical limitations such as inconsistent accessibility, incomplete historical coverage, or weak reproducibility.

The project shifted to **ENTSO-E** because it provided:

- standardized hourly electricity demand data
- reliable historical access
- consistent timestamp structure
- better reproducibility for forecasting experiments

This decision improved the stability and transparency of the full pipeline.

---

## Project Pipeline

The forecasting workflow consists of the following stages:

1. **Data ingestion**
   - ENTSO-E electricity load
   - NASA POWER weather variables

2. **Preprocessing**
   - hourly timestamp alignment
   - missing value handling
   - chronological consistency checks

3. **Feature engineering**
   - calendar features
   - cyclical encodings
   - lagged demand features
   - rolling statistics
   - weather covariates

4. **Horizon dataset construction**
   - `t_plus_1`
   - `t_plus_24`
   - `t_plus_168`

5. **Chronological splitting**
   - training
   - validation
   - test

6. **Model training**
   - Seasonal Naive
   - SARIMAX
   - Quantile GBR
   - Quantile GBR + Weather
   - DeepAR + Weather

7. **Evaluation**
   - RMSE
   - MAE
   - Pinball Loss
   - 80% interval coverage

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   ├── modeling/
│   ├── modeling_weather/
│   ├── horizons/
│   ├── horizons_weather/
│   ├── splits/
│   └── splits_weather/
│
├── models/
│   ├── horizon_quantile_gbr/
│   ├── horizon_quantile_gbr_weather/
│   └── horizon_deepar_weather/
│
├── reports/
│   ├── model_comparison_table.csv
│   ├── model_comparison_table.json
│   └── figures/
│       ├── rmse_vs_horizon.png
│       ├── interval_coverage.png
│       ├── forecast_vs_actual.png
│       └── pipeline_architecture.png
│
├── *.py
└── README.md

```
