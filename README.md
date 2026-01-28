# Ecommerce-RecSys

Traditional recommendation systems for e-commerce businesses using machine learning.

## Dataset

**Amazon Electronics Ratings Dataset**

| Metric | Value |
|--------|-------|
| Total Ratings | 7,824,482 |
| Unique Users | 4,201,696 |
| Unique Products | 476,002 |
| Time Period | Dec 1998 - Jul 2014 |
| Sparsity | 99.9996% |

### Rating Distribution

| Rating | Count | Percentage |
|--------|-------|------------|
| 5 stars | 4,347,541 | 55.6% |
| 4 stars | 1,485,781 | 19.0% |
| 3 stars | 633,073 | 8.1% |
| 2 stars | 456,322 | 5.8% |
| 1 star | 901,765 | 11.5% |

---

## Baseline Models

Baseline models capture inherent biases in rating data and serve as benchmarks for more sophisticated models.

### Model Descriptions

| Model | Formula | Description |
|-------|---------|-------------|
| **Global Mean** | μ | Predicts the overall average rating for all predictions |
| **User Bias** | μ + bᵤ | Accounts for users who rate higher/lower than average |
| **Item Bias** | μ + bᵢ | Accounts for items rated higher/lower than average |
| **User + Item Bias** | μ + bᵤ + bᵢ | Combined effect of user and item tendencies |

Where:
- **μ** = global mean rating (4.18)
- **bᵤ** = user bias (user's average - global average)
- **bᵢ** = item bias (item's average - global average)

### Results

**Data Filtering**: Users and items with ≥5 ratings (reduces cold start)
- Filtered dataset: 2,109,869 ratings (27% of original)
- Train/Test split: 80/20 (time-based)

| Model | RMSE | MAE | Notes |
|-------|------|-----|-------|
| Global Mean | **1.1757** | 0.9393 | Best RMSE |
| User Bias | 1.2735 | 0.9124 | |
| Item Bias | 1.1782 | 0.8687 | |
| User + Item Bias | 1.2494 | **0.8618** | Best MAE |

### Key Findings

1. **Global Mean wins RMSE**: The test set has 62% five-star ratings. Since the global mean (4.18) is close to 5, predicting the mean works well for the majority class.

2. **User+Item Bias wins MAE**: Makes better typical predictions (lower average error) but occasionally makes larger errors that inflate RMSE.

3. **Temporal Drift**: Time-based split causes user/item behaviors in test period to differ from training patterns.

### Error Analysis by Rating

| Actual Rating | Count | MAE | Observation |
|---------------|-------|-----|-------------|
| 5 stars | 263,564 | 0.64 | Slightly under-predicts |
| 4 stars | 78,345 | 0.63 | Good predictions |
| 3 stars | 34,681 | 1.15 | Moderate errors |
| 2 stars | 19,395 | 1.84 | Over-predicts |
| 1 star | 25,989 | 2.64 | Significantly over-predicts |

**Challenge**: Models struggle with negative ratings - they predict too high for 1-2 star items because training data is skewed positive.

### Coverage Analysis

- 90.2% of test users seen in training
- 84.2% of test items seen in training
- ~13K cold-start users in test set
- ~11K cold-start items in test set

---

## Matrix Factorization (SVD)

SVD decomposes the user-item matrix into latent factors, capturing hidden patterns in user preferences.

### Model Formula

```
r̂(u,i) = μ + bᵤ + bᵢ + pᵤ · qᵢ
```

Where:
- **μ** = global mean rating
- **bᵤ** = user bias (learned)
- **bᵢ** = item bias (learned)
- **pᵤ** = user latent factor vector (K dimensions)
- **qᵢ** = item latent factor vector (K dimensions)

### How It Works

```
User-Item Matrix (M × N)  ≈  User Factors (M × K)  ×  Item Factors (K × N)
     R (sparse)                    P              ×         Q^T
```

The model learns latent factors using Stochastic Gradient Descent (SGD), minimizing prediction error while regularizing to prevent overfitting.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent Factors (K) | 20 |
| Epochs | 20 |
| Learning Rate | 0.005 |
| Regularization | 0.02 |

### Results

| Model | RMSE | MAE | vs Baseline |
|-------|------|-----|-------------|
| Global Mean | 1.1757 | 0.9393 | baseline |
| User + Item Bias | 1.2494 | 0.8618 | -6.27% |
| **SVD (K=20)** | **1.1134** | **0.8412** | **+5.30%** |

### Key Findings

1. **SVD beats all baselines** - 5.30% RMSE improvement over Global Mean
2. **Best MAE** - 0.8412, better than all baseline models
3. **Training converged well** - RMSE dropped from 1.20 → 0.96 over 20 epochs

### Error Analysis by Rating (SVD)

| Actual Rating | Count | MAE | Bias |
|---------------|-------|-----|------|
| 5 stars | 263,564 | 0.66 | +0.66 |
| 4 stars | 78,345 | 0.34 | -0.19 |
| 3 stars | 34,681 | 1.10 | -1.09 |
| 2 stars | 19,395 | 2.02 | -2.02 |
| 1 star | 25,989 | 2.96 | -2.96 |

**Observation**: SVD still struggles with negative ratings due to positive skew in training data. The model tends to over-predict for low-rated items.

### Training Performance

- Training time: ~8 minutes (467s)
- Train RMSE progression: 1.20 → 0.96 (20 epochs)
- Model learns meaningful latent representations of user preferences and item characteristics

---

## Project Structure

```
Ecommerce-RecSys/
├── ratings_Electronics.csv   # Raw dataset (7.8M ratings)
├── eda_analysis.ipynb        # Exploratory Data Analysis notebook
├── baseline_models.py        # Baseline recommendation models
├── baseline_results.csv      # Baseline model results
├── svd_model.py              # Matrix Factorization (SVD) model
├── svd_results.csv           # SVD model results
├── EDA_Summary_Report.md     # Detailed EDA findings
├── Cold_Start_Problem.md     # Cold start problem explanation
└── README.md                 # This file
```

---

## Model Comparison Summary

| Model | RMSE | MAE | Notes |
|-------|------|-----|-------|
| Global Mean | 1.1757 | 0.9393 | Simplest baseline |
| User + Item Bias | 1.2494 | 0.8618 | Captures rating tendencies |
| **SVD (K=20)** | **1.1134** | **0.8412** | Best performance |

---

## Next Steps

- [x] Matrix Factorization (SVD)
- [ ] Neighborhood-based methods (User-KNN, Item-KNN)
- [ ] Neural Collaborative Filtering
- [ ] Hybrid approaches

---

## References

- Amazon Product Data: https://jmcauley.ucsd.edu/data/amazon/
