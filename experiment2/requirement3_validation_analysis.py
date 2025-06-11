import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os

def validate_and_analyze_clusters(validation_data_path="experiment2/validation_data.csv",
                                  optimal_k=2, # Determined from previous step
                                  output_dir="experiment2"): # Unused here, but for consistency
    """
    Loads preprocessed validation data, applies K-means with optimal k,
    calculates metrics, and analyzes cluster characteristics.
    """

    # --- 1. Load Data ---
    try:
        df_validation = pd.read_csv(validation_data_path)
        print(f"Successfully loaded validation data from '{validation_data_path}'. Shape: {df_validation.shape}")
    except FileNotFoundError:
        print(f"Error: Validation data file '{validation_data_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df_validation.empty:
        print("Error: Loaded validation data is empty.")
        return

    print(f"\nUsing optimal k = {optimal_k}")

    # --- 2. K-means Clustering on Validation Set ---
    try:
        kmeans_val = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42, algorithm='lloyd')
        validation_cluster_labels = kmeans_val.fit_predict(df_validation)
        print(f"\nSuccessfully applied K-means (k={optimal_k}) to the validation set.")
    except Exception as e:
        print(f"Error during K-means on validation set: {e}")
        return

    # --- 3. Calculate Internal Metrics on Validation Set ---
    try:
        val_silhouette_score = silhouette_score(df_validation, validation_cluster_labels)
        val_davies_bouldin_score = davies_bouldin_score(df_validation, validation_cluster_labels)

        print(f"\n--- Validation Set Metrics (k={optimal_k}) ---")
        print(f"Silhouette Score: {val_silhouette_score:.4f}")
        print(f"Davies-Bouldin Index: {val_davies_bouldin_score:.4f}")
    except Exception as e:
        print(f"Error calculating metrics on validation set: {e}")
        # Continue to analysis if labels were assigned
        if 'validation_cluster_labels' not in locals():
            return


    # --- 4. Analyze Clustering Results (using scaled features) ---
    df_validation_labeled = df_validation.copy()
    df_validation_labeled['Cluster'] = validation_cluster_labels

    print("\n--- Cluster Characteristics (Mean Scaled Feature Values) ---")
    # The data in validation_data.csv is already scaled.
    # These means represent the average scaled value (z-score) for each feature within each cluster.
    # A positive mean indicates the feature is, on average, above the overall dataset mean for that feature.
    # A negative mean indicates it's below the overall dataset mean.
    # The magnitude indicates how many standard deviations away from the mean.

    cluster_means = df_validation_labeled.groupby('Cluster').mean()
    print(cluster_means)

    print("\n--- Interpretation of Cluster Characteristics ---")
    # This interpretation is based on the scaled means.
    # For example, if 'review_overall' mean for Cluster 0 is positive and high,
    # it means beers in Cluster 0 tend to have higher overall reviews than average.
    # If 'ABV' mean for Cluster 1 is negative, beers in Cluster 1 tend to have lower ABV than average.

    # Generate comments based on the cluster_means DataFrame
    for i in range(optimal_k):
        print(f"\nCluster {i} Characteristics:")
        if i not in cluster_means.index:
            print(f"  Warning: Cluster {i} has no members or was not found in groupby results.")
            continue

        cluster_data = cluster_means.loc[i]
        for feature_name, mean_value in cluster_data.items():
            if mean_value > 0.5: # Arbitrary threshold for "notably higher"
                print(f"  - Notably higher average for {feature_name} (Mean Scaled: {mean_value:.2f})")
            elif mean_value < -0.5: # Arbitrary threshold for "notably lower"
                print(f"  - Notably lower average for {feature_name} (Mean Scaled: {mean_value:.2f})")
            else:
                # Values between -0.5 and 0.5 are closer to the overall average for that feature
                print(f"  - Average for {feature_name} is close to the overall mean (Mean Scaled: {mean_value:.2f})")

    # Example of more direct commentary (could be automated further based on thresholds)
    # Note: The features are: 'review_overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'ABV'
    # Cluster 0:
    # - If cluster_means.loc[0, 'review_overall'] is high (e.g. > 0.5), beers in this cluster tend to have high overall reviews.
    # - If cluster_means.loc[0, 'ABV'] is low (e.g. < -0.5), beers in this cluster tend to have lower ABV.
    # Cluster 1:
    # - Characteristics would be complementary or different based on its mean values.

    # This provides a quantitative basis for describing the clusters.
    # For instance, one cluster might represent highly-rated, higher ABV beers,
    # while the other might be average-rated, lower ABV beers. The actual interpretation
    # depends on the signs and magnitudes of the printed means.

if __name__ == "__main__":
    validate_and_analyze_clusters()
