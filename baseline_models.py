"""
Baseline Recommendation Models for E-commerce Ratings

This module implements simple but effective baseline models:
1. Global Mean
2. User Bias
3. Item Bias
4. User + Item Bias (Combined)

These serve as benchmarks for more sophisticated models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("=" * 70)
print("BASELINE RECOMMENDATION MODELS")
print("=" * 70)

print("\n[1/5] Loading data...")
start_time = time.time()

df = pd.read_csv('ratings_Electronics.csv',
                 names=['user_id', 'product_id', 'rating', 'timestamp'],
                 header=None)

print(f"      Loaded {len(df):,} ratings in {time.time()-start_time:.1f}s")

# ============================================================
# FILTER DATA (Reduce Cold Start)
# ============================================================

print("\n[2/5] Filtering cold start users/items...")

# Count interactions
user_counts = df['user_id'].value_counts()
item_counts = df['product_id'].value_counts()

# Filter: users with >=5 ratings, items with >=5 ratings
MIN_USER_RATINGS = 5
MIN_ITEM_RATINGS = 5

valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
valid_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index

df_filtered = df[df['user_id'].isin(valid_users) & df['product_id'].isin(valid_items)]

print(f"      Original: {len(df):,} ratings")
print(f"      Filtered: {len(df_filtered):,} ratings ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"      Users: {df_filtered['user_id'].nunique():,} (min {MIN_USER_RATINGS} ratings)")
print(f"      Items: {df_filtered['product_id'].nunique():,} (min {MIN_ITEM_RATINGS} ratings)")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

print("\n[3/5] Creating train/test split...")

# Time-based split: use last 20% of data for testing
df_filtered = df_filtered.sort_values('timestamp')
split_idx = int(len(df_filtered) * 0.8)

train_df = df_filtered.iloc[:split_idx]
test_df = df_filtered.iloc[split_idx:]

print(f"      Train set: {len(train_df):,} ratings ({len(train_df)/len(df_filtered)*100:.0f}%)")
print(f"      Test set:  {len(test_df):,} ratings ({len(test_df)/len(df_filtered)*100:.0f}%)")

# ============================================================
# CALCULATE BIASES FROM TRAINING DATA
# ============================================================

print("\n[4/5] Calculating biases from training data...")

# Global mean
global_mean = train_df['rating'].mean()
print(f"      Global Mean (mu): {global_mean:.4f}")

# User biases: user_avg - global_mean
user_means = train_df.groupby('user_id')['rating'].mean()
user_bias = user_means - global_mean
print(f"      User biases calculated for {len(user_bias):,} users")
print(f"      User bias range: [{user_bias.min():.2f}, {user_bias.max():.2f}]")

# Item biases: item_avg - global_mean
item_means = train_df.groupby('product_id')['rating'].mean()
item_bias = item_means - global_mean
print(f"      Item biases calculated for {len(item_bias):,} items")
print(f"      Item bias range: [{item_bias.min():.2f}, {item_bias.max():.2f}]")

# ============================================================
# BASELINE MODELS - PREDICTIONS
# ============================================================

print("\n[5/5] Making predictions on test set...")

# Get test data
test_users = test_df['user_id'].values
test_items = test_df['product_id'].values
y_true = test_df['rating'].values

# --- Model 1: Global Mean ---
pred_global_mean = np.full(len(test_df), global_mean)

# --- Model 2: User Bias ---
# For users not in training, use 0 bias
user_bias_values = test_df['user_id'].map(user_bias).fillna(0).values
pred_user_bias = global_mean + user_bias_values

# --- Model 3: Item Bias ---
# For items not in training, use 0 bias
item_bias_values = test_df['product_id'].map(item_bias).fillna(0).values
pred_item_bias = global_mean + item_bias_values

# --- Model 4: User + Item Bias ---
pred_combined = global_mean + user_bias_values + item_bias_values

# Clip predictions to valid range [1, 5]
pred_global_mean = np.clip(pred_global_mean, 1, 5)
pred_user_bias = np.clip(pred_user_bias, 1, 5)
pred_item_bias = np.clip(pred_item_bias, 1, 5)
pred_combined = np.clip(pred_combined, 1, 5)

# ============================================================
# EVALUATE MODELS
# ============================================================

def evaluate_model(y_true, y_pred, model_name):
    """Calculate RMSE and MAE for a model."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae}

results = []
results.append(evaluate_model(y_true, pred_global_mean, 'Global Mean'))
results.append(evaluate_model(y_true, pred_user_bias, 'User Bias'))
results.append(evaluate_model(y_true, pred_item_bias, 'Item Bias'))
results.append(evaluate_model(y_true, pred_combined, 'User + Item Bias'))

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS: BASELINE MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df['RMSE'] = results_df['RMSE'].round(4)
results_df['MAE'] = results_df['MAE'].round(4)

# Calculate improvement over global mean
baseline_rmse = results_df[results_df['Model'] == 'Global Mean']['RMSE'].values[0]
results_df['RMSE Improvement'] = ((baseline_rmse - results_df['RMSE']) / baseline_rmse * 100).round(2)
results_df['RMSE Improvement'] = results_df['RMSE Improvement'].apply(lambda x: f"{x:+.2f}%" if x != 0 else "baseline")

print("\n" + results_df.to_string(index=False))

# Best model
best_model = results_df.loc[results_df['RMSE'].idxmin()]
print(f"\nBest Baseline Model: {best_model['Model']}")
print(f"  RMSE: {best_model['RMSE']:.4f}")
print(f"  MAE:  {best_model['MAE']:.4f}")

# ============================================================
# DETAILED ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

# Error distribution for best model (User + Item Bias)
errors = y_true - pred_combined
print(f"\nUser + Item Bias Model - Error Analysis:")
print(f"  Mean Error (bias):     {errors.mean():.4f}")
print(f"  Std Dev of Errors:     {errors.std():.4f}")
print(f"  Min Error:             {errors.min():.2f}")
print(f"  Max Error:             {errors.max():.2f}")

# Error by rating value
print(f"\nError breakdown by actual rating:")
for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
    mask = y_true == rating
    if mask.sum() > 0:
        rating_errors = errors[mask]
        print(f"  Rating {int(rating)}: MAE = {np.abs(rating_errors).mean():.3f}, "
              f"Mean Error = {rating_errors.mean():+.3f}, Count = {mask.sum():,}")

# Coverage analysis
print(f"\nCoverage Analysis:")
users_in_test = set(test_df['user_id'])
users_in_train = set(train_df['user_id'])
items_in_test = set(test_df['product_id'])
items_in_train = set(train_df['product_id'])

cold_users = users_in_test - users_in_train
cold_items = items_in_test - items_in_train

print(f"  Test users seen in training: {len(users_in_test & users_in_train):,} / {len(users_in_test):,} "
      f"({len(users_in_test & users_in_train)/len(users_in_test)*100:.1f}%)")
print(f"  Test items seen in training: {len(items_in_test & items_in_train):,} / {len(items_in_test):,} "
      f"({len(items_in_test & items_in_train)/len(items_in_test)*100:.1f}%)")
print(f"  Cold start users in test: {len(cold_users):,}")
print(f"  Cold start items in test: {len(cold_items):,}")

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv('baseline_results.csv', index=False)
print(f"\nResults saved to: baseline_results.csv")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
What the baseline models tell us:

1. GLOBAL MEAN: Simply predicting the average rating (4.01) gives us
   a starting point. Any useful model must beat this.

2. USER BIAS: Some users are lenient (rate everything high), others are
   strict. Capturing this improves predictions.

3. ITEM BIAS: Some products are genuinely better and receive higher
   ratings. This is valuable signal.

4. COMBINED: User + Item bias together captures both effects. This is
   typically the strongest baseline and often hard to beat significantly
   with complex models.

Next Steps:
- Matrix Factorization (SVD) should improve RMSE by 5-15%
- Neighborhood methods (KNN) offer interpretability
- Deep learning methods may provide marginal gains
""")
