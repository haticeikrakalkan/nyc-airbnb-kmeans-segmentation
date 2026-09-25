# NYC Airbnb Market Segmentation with K-Means

Unsupervised clustering project that segments New York City Airbnb listings into meaningful 
market groups based on pricing, location, availability, and review activity.

## Overview

This project applies K-Means clustering to the [NYC Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) 
dataset (48,895 listings, 16 features) to uncover natural segments in the listings — without 
using any labels. The resulting clusters reveal distinct listing archetypes, from actively 
managed popular listings to dormant, rarely-reviewed ones.

## Workflow

1. **Data cleaning** — dropped identifier/free-text columns (`id`, `name`, `host_id`, `host_name`), 
   handled missing values, engineered `days_since_last_review` from the raw date column
2. **Outlier handling & log transform** — cleaned extreme/invalid `price` values and applied 
   `log1p` to correct for skewness across several numeric features
3. **Exploratory Data Analysis** — distribution plots, category counts, and a correlation heatmap
4. **Preprocessing** — one-hot encoding for categorical features, `StandardScaler` for numeric 
   features via `ColumnTransformer`
5. **Model selection** — determined the optimal number of clusters using the Elbow Method 
   (`KneeLocator`) and Silhouette Score
6. **Clustering** — trained the final K-Means model (`k=4`) and profiled each cluster
7. **Validation** — verified train/test consistency and stability across different random seeds

## A key debugging moment

An early version of the pipeline accidentally applied `StandardScaler` to one-hot encoded 
columns. This inflated the influence of the rare `Staten Island` category so much that K-means 
was effectively just splitting listings into "Staten Island vs. not" rather than finding a 
meaningful segmentation (visible as an anomalously high silhouette score at `k=2`). This was 
diagnosed by inspecting `cluster_centers_`, and fixed by scaling only continuous numeric columns. 
This is documented in the notebook as a worked example of debugging a K-means pipeline.

## Results

Four clusters emerged from the final model:

| Cluster | Name | Key traits |
|---|---|---|
| 0 | Mid-tier, low availability | Rarely vacant, Manhattan/Brooklyn mix |
| 1 | Premium / Professional host | Expensive, multi-listing hosts, Manhattan-heavy |
| 2 | Passive / dormant listings | Almost no reviews, inactive for a long time |
| 3 | Popular / active listings | Most-reviewed, fresh activity, Brooklyn-leaning, largest segment |

Clusters were validated for:
- **Consistency** — nearly identical cluster proportions between train and test sets
- **Stability** — silhouette scores remained in a tight band (0.15–0.18) across five different 
  `random_state` values

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (`KMeans`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `silhouette_score`)
- kneed (`KneeLocator` for automatic elbow detection)
- matplotlib, seaborn, plotly (visualization)

## Project Structure
├── NYC_Airbnb_KMeans_Segmentation.ipynb # Full analysis notebook
├── airbnb.csv # Dataset (not included — see below)
└── README.md


## Dataset

The dataset is not included in this repo due to size/licensing. Download it from 
[Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) and place 
`airbnb.csv` in the project root before running the notebook.

## Getting Started

```bash
pip install pandas numpy scikit-learn kneed matplotlib seaborn plotly
jupyter notebook NYC_Airbnb_KMeans_Segmentation.ipynb
```

## Possible Next Steps

- Compare against density-based clustering (DBSCAN)
- Try alternative values of `k` for finer-grained business segments
- Build a price-prediction model conditioned on cluster membership
