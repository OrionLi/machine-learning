import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os

def find_optimal_k(data_path="experiment2/train_data.csv", output_dir="experiment2"):
    """
    Loads preprocessed data, runs K-means for a range of k values,
    calculates Silhouette and Davies-Bouldin scores, plots them,
    and suggests an optimal k.
    """
    # Ensure output directory exists (it should from previous script)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- 1. Load Data ---
    try:
        df_train = pd.read_csv(data_path)
        print(f"Successfully loaded training data from '{data_path}'. Shape: {df_train.shape}")
    except FileNotFoundError:
        print(f"Error: Training data file '{data_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df_train.empty:
        print("Error: Loaded training data is empty.")
        return

    # --- 2. K-means and Metrics Calculation ---
    k_range = range(2, 16)  # Test k from 2 to 15
    silhouette_scores = []
    davies_bouldin_scores = []

    print(f"\nCalculating K-means metrics for k in {list(k_range)}...")

    for k_val in k_range:
        try:
            kmeans = KMeans(n_clusters=k_val, n_init='auto', random_state=42, algorithm='lloyd') # Explicitly set algorithm
            cluster_labels = kmeans.fit_predict(df_train) # Fit and get labels

            # Silhouette Score (requires at least 2 labels and more than 1 cluster)
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(df_train, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(np.nan) # Should not happen with k_val >= 2
                print(f"Warning: Only one cluster found for k={k_val}. Silhouette score set to NaN.")

            # Davies-Bouldin Index (requires at least 2 labels)
            if len(np.unique(cluster_labels)) > 1:
                db_score = davies_bouldin_score(df_train, cluster_labels)
                davies_bouldin_scores.append(db_score)
            else:
                davies_bouldin_scores.append(np.nan) # Should not happen
                print(f"Warning: Only one cluster found for k={k_val}. Davies-Bouldin score set to NaN.")

            print(f"k={k_val}: Silhouette={sil_score:.4f}, Davies-Bouldin={db_score:.4f}")

        except Exception as e:
            print(f"Error during K-means for k={k_val}: {e}")
            silhouette_scores.append(np.nan)
            davies_bouldin_scores.append(np.nan)

    # --- 3. Plotting Metrics ---
    # Plot Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
    plt.title("Silhouette Score vs. Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(list(k_range))
    plt.grid(True)
    silhouette_plot_path = os.path.join(output_dir, "silhouette_scores.png")
    try:
        plt.savefig(silhouette_plot_path)
        print(f"\nSilhouette score plot saved to '{silhouette_plot_path}'")
    except Exception as e:
        print(f"Error saving silhouette plot: {e}")
    plt.close()

    # Plot Davies-Bouldin Index
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, davies_bouldin_scores, marker='o', linestyle='-')
    plt.title("Davies-Bouldin Index vs. Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies-Bouldin Index")
    plt.xticks(list(k_range))
    plt.grid(True)
    db_plot_path = os.path.join(output_dir, "davies_bouldin_scores.png")
    try:
        plt.savefig(db_plot_path)
        print(f"Davies-Bouldin Index plot saved to '{db_plot_path}'")
    except Exception as e:
        print(f"Error saving Davies-Bouldin plot: {e}")
    plt.close()

    # --- 4. Determine Optimal k ---
    print("\n--- Metric Values ---")
    for i, k_val in enumerate(k_range):
        print(f"k={k_val}: Silhouette Score = {silhouette_scores[i]:.4f}, Davies-Bouldin Index = {davies_bouldin_scores[i]:.4f}")

    # Logic for choosing optimal k:
    # Maximize Silhouette Score (closer to 1 is better)
    # Minimize Davies-Bouldin Index (closer to 0 is better)
    # Look for an "elbow" or significant change.

    optimal_k_silhouette = -1
    max_silhouette_score = -np.inf # Initialize with a very small number
    if all(np.isnan(s) for s in silhouette_scores):
        print("Warning: All Silhouette scores are NaN. Cannot determine optimal k from Silhouette.")
    else:
        max_silhouette_score = np.nanmax(silhouette_scores)
        optimal_k_silhouette = k_range[np.nanargmax(silhouette_scores)]

    optimal_k_db = -1
    min_db_score = np.inf # Initialize with a very large number
    if all(np.isnan(d) for d in davies_bouldin_scores):
        print("Warning: All Davies-Bouldin scores are NaN. Cannot determine optimal k from Davies-Bouldin.")
    else:
        min_db_score = np.nanmin(davies_bouldin_scores)
        optimal_k_db = k_range[np.nanargmin(davies_bouldin_scores)]

    print(f"\n--- Optimal k Suggestion ---")
    print(f"Based on Silhouette Score: Optimal k = {optimal_k_silhouette} (Score: {max_silhouette_score:.4f})")
    print(f"Based on Davies-Bouldin Index: Optimal k = {optimal_k_db} (Score: {min_db_score:.4f})")

    # Final decision often involves looking at the plots for elbows or significant changes,
    # and considering if the k values from both metrics align or make sense for the problem domain.
    # For this script, we'll state a preference or a common choice if they differ.
    # If they are the same, that's a strong candidate.
    chosen_k = -1
    if optimal_k_silhouette != -1 and optimal_k_db != -1:
        if optimal_k_silhouette == optimal_k_db:
            chosen_k = optimal_k_silhouette
            print(f"\nBoth metrics suggest k = {chosen_k} as optimal.")
        else:
            # If they differ, a common approach is to prioritize, or look for a k where
            # Silhouette is high and DB is reasonably low, possibly favoring simpler models (smaller k)
            # or looking for a "knee" in DB plot if Silhouette plateaus.
            # For now, let's say we might favor the one from Silhouette if it's clear,
            # or a balance. Here, we might just report both and suggest user inspects plots.
            print(f"\nMetrics suggest different optimal k values (Silhouette k={optimal_k_silhouette}, Davies-Bouldin k={optimal_k_db}).")
            print("Inspect the plots: ")
            print(f"  - Silhouette plot ('{silhouette_plot_path}') for a peak.")
            print(f"  - Davies-Bouldin plot ('{db_plot_path}') for a valley or 'elbow'.")
            print("Consider the trade-off or if one metric is more critical for your specific application.")
            # As a simple heuristic for this script, let's pick the one suggested by Silhouette,
            # as its interpretation (separation and cohesion) is often intuitive.
            chosen_k = optimal_k_silhouette
            print(f"For this automated suggestion, we will tentatively select k = {chosen_k} based on the maximum Silhouette Score.")

    elif optimal_k_silhouette != -1:
        chosen_k = optimal_k_silhouette
        print(f"\nOptimal k based on Silhouette Score: {chosen_k} (Davies-Bouldin was inconclusive).")
    elif optimal_k_db != -1:
        chosen_k = optimal_k_db
        print(f"\nOptimal k based on Davies-Bouldin Index: {chosen_k} (Silhouette was inconclusive).")
    else:
        print("\nCould not determine an optimal k from the calculated metrics.")

    print(f"\nChosen optimal k for reporting: {chosen_k}")
    # This chosen_k can be written to a file or used in a subsequent script.

if __name__ == "__main__":
    # KMeans with n_init='auto' might issue a UserWarning about future changes in default behavior.
    # This is normal and doesn't affect the current execution.
    # The default 'lloyd' algorithm is specified to avoid another UserWarning regarding algorithm choice.
    find_optimal_k()
