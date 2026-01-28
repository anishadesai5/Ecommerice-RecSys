"""
Matrix Factorization (SVD) for Recommendation Systems

This module implements SVD-based collaborative filtering using
Stochastic Gradient Descent (SGD) optimization.

Model: r_hat(u,i) = mu + b_u + b_i + p_u . q_i

Where:
  - mu: global mean rating
  - b_u: user bias
  - b_i: item bias
  - p_u: user latent factor vector
  - q_i: item latent factor vector
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from collections import defaultdict

# ============================================================
# SVD MODEL CLASS
# ============================================================

class SVD:
    """
    Matrix Factorization model using Stochastic Gradient Descent.

    Parameters:
    -----------
    n_factors : int
        Number of latent factors (default: 20)
    n_epochs : int
        Number of training iterations (default: 20)
    lr : float
        Learning rate (default: 0.005)
    reg : float
        Regularization parameter (default: 0.02)
    verbose : bool
        Print training progress (default: True)
    """

    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, verbose=True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.verbose = verbose

    def fit(self, train_data):
        """
        Train the SVD model.

        Parameters:
        -----------
        train_data : DataFrame
            Training data with columns: user_id, product_id, rating
        """
        # Create mappings from IDs to indices
        self.user_ids = train_data['user_id'].unique()
        self.item_ids = train_data['product_id'].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        if self.verbose:
            print(f"    Users: {n_users:,}, Items: {n_items:,}")
            print(f"    Factors: {self.n_factors}, LR: {self.lr}, Reg: {self.reg}")

        # Global mean
        self.global_mean = train_data['rating'].mean()

        # Initialize biases to zero
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Initialize latent factors with small random values
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))  # User factors
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))  # Item factors

        # Convert training data to arrays for faster access
        users = train_data['user_id'].map(self.user_to_idx).values
        items = train_data['product_id'].map(self.item_to_idx).values
        ratings = train_data['rating'].values

        # Training loop
        if self.verbose:
            print(f"\n    Training for {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            # Shuffle training data
            indices = np.random.permutation(len(ratings))

            total_error = 0
            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                # Predict
                pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
                pred += np.dot(self.P[u], self.Q[i])

                # Calculate error
                error = r - pred
                total_error += error ** 2

                # Update biases
                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])

                # Update latent factors
                P_u = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u - self.reg * self.Q[i])

            rmse = np.sqrt(total_error / len(ratings))
            if self.verbose:
                print(f"      Epoch {epoch+1:2d}/{self.n_epochs}: Train RMSE = {rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predict rating for a single user-item pair."""
        # Handle unknown users/items
        if user_id not in self.user_to_idx:
            if item_id not in self.item_to_idx:
                return self.global_mean
            else:
                i = self.item_to_idx[item_id]
                return self.global_mean + self.item_bias[i]

        if item_id not in self.item_to_idx:
            u = self.user_to_idx[user_id]
            return self.global_mean + self.user_bias[u]

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]

        pred = self.global_mean + self.user_bias[u] + self.item_bias[i]
        pred += np.dot(self.P[u], self.Q[i])

        # Clip to valid rating range
        return np.clip(pred, 1.0, 5.0)

    def predict_batch(self, test_data):
        """Predict ratings for a batch of user-item pairs."""
        predictions = []
        for _, row in test_data.iterrows():
            pred = self.predict(row['user_id'], row['product_id'])
            predictions.append(pred)
        return np.array(predictions)

    def get_user_factors(self, user_id):
        """Get latent factors for a user."""
        if user_id in self.user_to_idx:
            u = self.user_to_idx[user_id]
            return self.P[u]
        return None

    def get_item_factors(self, item_id):
        """Get latent factors for an item."""
        if item_id in self.item_to_idx:
            i = self.item_to_idx[item_id]
            return self.Q[i]
        return None

    def recommend_items(self, user_id, n=10, exclude_known=True, known_items=None):
        """
        Recommend top-N items for a user.

        Parameters:
        -----------
        user_id : str
            User ID
        n : int
            Number of recommendations
        exclude_known : bool
            Whether to exclude items the user has already rated
        known_items : set
            Set of item IDs the user has already interacted with
        """
        if user_id not in self.user_to_idx:
            # Cold start: return most popular items (by average item bias)
            top_items = np.argsort(self.item_bias)[::-1][:n]
            return [self.item_ids[i] for i in top_items]

        u = self.user_to_idx[user_id]

        # Calculate scores for all items
        scores = self.global_mean + self.user_bias[u] + self.item_bias
        scores += np.dot(self.Q, self.P[u])

        # Get top items
        if exclude_known and known_items:
            known_indices = {self.item_to_idx[iid] for iid in known_items
                          if iid in self.item_to_idx}
            for idx in known_indices:
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        return [(self.item_ids[i], scores[i]) for i in top_indices]


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MATRIX FACTORIZATION (SVD) MODEL")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. LOAD DATA
    # ----------------------------------------------------------
    print("\n[1/6] Loading data...")
    start_time = time.time()

    df = pd.read_csv('ratings_Electronics.csv',
                     names=['user_id', 'product_id', 'rating', 'timestamp'],
                     header=None)

    print(f"      Loaded {len(df):,} ratings in {time.time()-start_time:.1f}s")

    # ----------------------------------------------------------
    # 2. FILTER DATA
    # ----------------------------------------------------------
    print("\n[2/6] Filtering cold start users/items...")

    MIN_USER_RATINGS = 5
    MIN_ITEM_RATINGS = 5

    user_counts = df['user_id'].value_counts()
    item_counts = df['product_id'].value_counts()

    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    valid_items = item_counts[item_counts >= MIN_ITEM_RATINGS].index

    df_filtered = df[df['user_id'].isin(valid_users) & df['product_id'].isin(valid_items)]

    print(f"      Original: {len(df):,} ratings")
    print(f"      Filtered: {len(df_filtered):,} ratings ({len(df_filtered)/len(df)*100:.1f}%)")

    # ----------------------------------------------------------
    # 3. TRAIN/TEST SPLIT
    # ----------------------------------------------------------
    print("\n[3/6] Creating train/test split (time-based 80/20)...")

    df_filtered = df_filtered.sort_values('timestamp')
    split_idx = int(len(df_filtered) * 0.8)

    train_df = df_filtered.iloc[:split_idx].copy()
    test_df = df_filtered.iloc[split_idx:].copy()

    print(f"      Train set: {len(train_df):,} ratings")
    print(f"      Test set:  {len(test_df):,} ratings")

    # ----------------------------------------------------------
    # 4. TRAIN SVD MODEL
    # ----------------------------------------------------------
    print("\n[4/6] Training SVD model...")
    print("-" * 50)

    # Hyperparameters
    N_FACTORS = 20      # Number of latent factors
    N_EPOCHS = 20       # Training iterations
    LEARNING_RATE = 0.005
    REGULARIZATION = 0.02

    model = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        reg=REGULARIZATION,
        verbose=True
    )

    train_start = time.time()
    model.fit(train_df)
    train_time = time.time() - train_start

    print("-" * 50)
    print(f"    Training completed in {train_time:.1f}s")

    # ----------------------------------------------------------
    # 5. EVALUATE MODEL
    # ----------------------------------------------------------
    print("\n[5/6] Evaluating on test set...")

    eval_start = time.time()

    # Predict in batches for speed
    print("      Making predictions...")
    y_true = test_df['rating'].values
    y_pred = []

    # Process in chunks for memory efficiency
    chunk_size = 10000
    for i in range(0, len(test_df), chunk_size):
        chunk = test_df.iloc[i:i+chunk_size]
        for _, row in chunk.iterrows():
            pred = model.predict(row['user_id'], row['product_id'])
            y_pred.append(pred)
        if (i + chunk_size) % 50000 == 0:
            print(f"      Processed {min(i+chunk_size, len(test_df)):,} / {len(test_df):,}")

    y_pred = np.array(y_pred)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    eval_time = time.time() - eval_start
    print(f"      Evaluation completed in {eval_time:.1f}s")

    # ----------------------------------------------------------
    # 6. RESULTS
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
    SVD Model Performance:
    ----------------------
    RMSE: {rmse:.4f}
    MAE:  {mae:.4f}

    Hyperparameters:
    ----------------
    Latent Factors (K): {N_FACTORS}
    Epochs:             {N_EPOCHS}
    Learning Rate:      {LEARNING_RATE}
    Regularization:     {REGULARIZATION}

    Comparison with Baselines:
    --------------------------
    Model               RMSE      MAE       RMSE Improvement
    -----               ----      ---       ----------------
    Global Mean         1.1757    0.9393    (baseline)
    User + Item Bias    1.2494    0.8618    -6.27%
    SVD (K={N_FACTORS})           {rmse:.4f}    {mae:.4f}    {((1.1757-rmse)/1.1757)*100:+.2f}%
    """)

    # Error analysis
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    errors = y_true - y_pred
    print(f"""
    Error Distribution:
    -------------------
    Mean Error:    {errors.mean():.4f}
    Std Dev:       {errors.std():.4f}
    Min Error:     {errors.min():.2f}
    Max Error:     {errors.max():.2f}
    """)

    print("    Error by Actual Rating:")
    print("    " + "-" * 45)
    for rating in [1.0, 2.0, 3.0, 4.0, 5.0]:
        mask = y_true == rating
        if mask.sum() > 0:
            rating_mae = np.abs(errors[mask]).mean()
            rating_mean_error = errors[mask].mean()
            print(f"    Rating {int(rating)}: MAE = {rating_mae:.3f}, "
                  f"Bias = {rating_mean_error:+.3f}, Count = {mask.sum():,}")

    # ----------------------------------------------------------
    # SAMPLE RECOMMENDATIONS
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 70)

    # Get a sample user with multiple ratings
    user_rating_counts = train_df['user_id'].value_counts()
    sample_users = user_rating_counts[user_rating_counts >= 10].head(3).index.tolist()

    for user_id in sample_users:
        user_items = set(train_df[train_df['user_id'] == user_id]['product_id'])
        recommendations = model.recommend_items(user_id, n=5,
                                                exclude_known=True,
                                                known_items=user_items)

        print(f"\n    User: {user_id}")
        print(f"    Previously rated: {len(user_items)} items")
        print(f"    Top 5 Recommendations:")
        for i, (item_id, score) in enumerate(recommendations, 1):
            print(f"      {i}. {item_id} (predicted rating: {score:.2f})")

    # ----------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------
    results = {
        'model': 'SVD',
        'n_factors': N_FACTORS,
        'n_epochs': N_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'regularization': REGULARIZATION,
        'rmse': rmse,
        'mae': mae,
        'train_time_seconds': train_time
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv('svd_results.csv', index=False)
    print(f"\n    Results saved to: svd_results.csv")

    print("\n" + "=" * 70)
    print("SVD MODEL COMPLETE")
    print("=" * 70)
