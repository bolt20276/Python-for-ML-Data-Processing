# Python for ML Data Processing

Getting an E-Commerce dataset ML-ready for churn prediction — built on Microsoft Fabric using Polars, with no modelling. The output is a clean, feature-engineered, stratified Parquet dataset ready for a downstream model.


## Architecture

```
Kaggle CSV
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               MICROSOFT FABRIC LAKEHOUSE             │
│                                                     │
│  BRONZE              SILVER              GOLD       │
│  Files/bronze/  ──▶  Files/silver/  ──▶  Tables/   │
│                                                     │
│  Raw ingest          Clean                Split     │
│  Parquet snapshot    Feature engineer     Store     │
└─────────────────────────────────────────────────────┘
```

| Layer | Path | What happens |
|---|---|---|
| Bronze | `Files/bronze/` | Raw CSV ingested, Parquet snapshot — no transforms |
| Silver | `Files/silver/` | Cleaning, dtype casting, feature engineering, split |
| Gold | `Tables/` | ML-ready Delta tables — queryable via SQL and Power BI |

---

## Pipeline Notebooks

| # | Notebook | Layer | Description |
|---|---|---|---|
| 1 | `01_ingest_bronze.ipynb` | Bronze | Download from Kaggle, write raw Parquet |
| 2 | `02_clean_silver.ipynb` | Silver | Null imputation, dtype casting, deduplication |
| 3 | `03_feature_engineering.ipynb` | Silver | RFM features, risk score, categorical encoding |
| 4 | `04_split_and_save_gold.ipynb` | Silver → Gold | Stratified 80/20 split, write Delta tables |

---

## Key Design Decisions

- **Polars over Pandas** — lazy evaluation, strict types, multi-core. Up to 400× faster on vectorized operations vs Python loops
- **Vectorized expressions** — zero for-loops in the pipeline. All transforms use `pl.with_columns()` and `pl.when().then().otherwise()`
- **Dtype downcasting** — `Float64→Float32`, `Utf8→Categorical` — ~65% memory reduction in one pass
- **Stratified split** — churn label is ~17% positive. Stratification preserves that ratio in both train and test sets

## Dataset

[E-Commerce Customer Churn — Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
· ~5,630 customers · 20 features · ~17% churn rate

## Tech Stack

`Microsoft Fabric` · `Polars` · `kagglehub` · `Delta Lake` · `scikit-learn` · `Python 3.10+`

## Setup

Add to your Fabric Environment (no `%pip install` in notebooks):
polars>=1.0.0
scikit-learn>=1.3.0
kagglehub>=0.3.0

Set Kaggle credentials before running notebook 1:
```python
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"]      = "your_api_key"

---

*Data preparation stage only — modelling is handled separately.*
