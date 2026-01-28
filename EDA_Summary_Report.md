# EDA Summary Report: Amazon Electronics Ratings Dataset

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Ratings** | 7,824,482 |
| **Unique Users** | 4,201,696 |
| **Unique Products** | 476,002 |
| **Time Period** | Dec 1998 - Jul 2014 |
| **File Size** | 304 MB |

---

## Data Quality

- **Missing Values**: 0 (No missing data)
- **Duplicate Records**: 0
- **Data Completeness**: 100%

---

## Rating Distribution

| Rating | Count | Percentage |
|--------|-------|------------|
| 5 stars | 4,347,541 | 55.6% |
| 4 stars | 1,485,781 | 19.0% |
| 3 stars | 633,073 | 8.1% |
| 2 stars | 456,322 | 5.8% |
| 1 star | 901,765 | 11.5% |

### Key Observations
- **Strong positive bias**: 74.6% of ratings are 4 or 5 stars
- **Mean Rating**: 4.012
- **Median Rating**: 5.0
- **Standard Deviation**: 1.381
- The J-shaped distribution is typical for e-commerce platforms where satisfied customers are more likely to leave reviews

---

## Sparsity Analysis

| Metric | Value |
|--------|-------|
| Matrix Dimensions | 4,201,696 × 476,002 |
| Total Possible Interactions | ~2 trillion |
| Actual Interactions | 7,824,482 |
| **Density** | 0.000391% |
| **Sparsity** | 99.9996% |

### Implications
- Extremely sparse matrix typical for recommendation systems
- Matrix factorization methods (SVD, ALS) are well-suited
- Memory-efficient sparse matrix representations required

---

## User Activity Analysis

| Metric | Value |
|--------|-------|
| Min ratings per user | 1 |
| Max ratings per user | 520 |
| Mean ratings per user | 1.86 |
| Median ratings per user | 1 |

### User Segments

| Segment | Count | Percentage |
|---------|-------|------------|
| 1 rating (one-time users) | 2,881,832 | 68.6% |
| 2-5 ratings | ~1,000,000 | ~24% |
| 6-10 ratings | ~200,000 | ~5% |
| 11-50 ratings | ~100,000 | ~2% |
| 50+ ratings (power users) | ~20,000 | <1% |

### Key Finding
- **68.6% of users have only 1 rating** - severe cold start challenge
- Long-tail distribution: few power users, many casual users

---

## Product Popularity Analysis

| Metric | Value |
|--------|-------|
| Min ratings per product | 1 |
| Max ratings per product | 18,244 |
| Mean ratings per product | 16.44 |
| Median ratings per product | 2 |

### Product Segments

| Segment | Count | Percentage |
|---------|-------|------------|
| 1 rating only | 179,738 | 37.8% |
| 2-5 ratings | ~150,000 | ~32% |
| 6-20 ratings | ~80,000 | ~17% |
| 21-100 ratings | ~50,000 | ~10% |
| 100+ ratings (popular) | ~16,000 | ~3% |

### Key Finding
- **37.8% of products have only 1 rating**
- Classic long-tail distribution in e-commerce

---

## Temporal Analysis

### Time Range
- **Earliest Rating**: December 4, 1998
- **Latest Rating**: July 23, 2014
- **Duration**: 5,710 days (~15.6 years)

### Yearly Trends
- Rating volume increased significantly over time
- Most data concentrated in 2012-2014
- Average rating remained relatively stable (3.9-4.1) across years

### Day of Week Patterns
- Relatively uniform distribution across weekdays
- Slight variations in average rating by day (within 4.0-4.1 range)

---

## Cold Start Problem Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Users with ≤1 rating | 2,881,832 | 68.6% |
| Users with ≤5 ratings | ~3,900,000 | ~93% |
| Products with ≤1 rating | 179,738 | 37.8% |
| Products with ≤5 ratings | ~330,000 | ~69% |

---

## Recommendations for Modeling

### 1. Data Preprocessing
- Filter users with minimum 5+ interactions
- Filter products with minimum 5+ ratings
- This will significantly reduce cold start issues but also reduce dataset size

### 2. Recommended Models
- **Baseline**: Global mean, user/item bias models
- **Collaborative Filtering**: Matrix Factorization (SVD, ALS, NMF)
- **Neighborhood Methods**: User-based or Item-based KNN
- **Deep Learning**: Neural Collaborative Filtering (if compute available)

### 3. Evaluation Strategy
- Use **time-based train/test split** (not random) to simulate real-world scenario
- Metrics: RMSE, MAE for rating prediction; Precision@K, Recall@K, NDCG for ranking

### 4. Cold Start Mitigation
- Popularity-based fallback for new users/items
- Content-based methods if product metadata available
- Hybrid approaches combining multiple signals

---

## Files in Repository

| File | Description |
|------|-------------|
| `ratings_Electronics.csv` | Raw dataset (7.8M ratings) |
| `eda_analysis.ipynb` | Jupyter notebook with full EDA code |
| `EDA_Summary_Report.md` | This summary report |
| `Cold_Start_Problem.md` | Explanation of cold start in RecSys |
