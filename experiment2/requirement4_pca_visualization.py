import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score # For reference in comments
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import os

def visualize_pca_clusters(validation_data_path="experiment2/validation_data.csv",
                             optimal_k=2, # Determined from previous steps
                             output_dir="experiment2"):
    """
    Loads preprocessed validation data, re-runs K-means, applies PCA,
    and visualizes the clusters in 2D PCA space.
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

    # --- 2. Re-run K-means to get cluster labels ---
    # This ensures consistency if this script is run independently or if data access changes.
    try:
        kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=42, algorithm='lloyd')
        cluster_labels = kmeans.fit_predict(df_validation)
        print(f"\nSuccessfully applied K-means (k={optimal_k}) to get cluster labels for PCA visualization.")
    except Exception as e:
        print(f"Error during K-means for PCA visualization: {e}")
        return

    # --- 3. Apply PCA ---
    try:
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(df_validation)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = cluster_labels
        print(f"\nSuccessfully applied PCA, reducing data to 2 components. Explained variance ratio: {pca.explained_variance_ratio_}")
    except Exception as e:
        print(f"Error during PCA: {e}")
        return

    # --- 4. Create Scatter Plot ---
    plt.figure(figsize=(10, 7))

    # Define colors for clusters (can be extended if optimal_k > 2)
    colors = ['royalblue', 'orangered', 'forestgreen', 'gold', 'purple', 'saddlebrown']
    if optimal_k > len(colors): # Fallback if more clusters than predefined colors
        # Generate random colors, less ideal for fixed interpretation but works
        import random
        rng = random.Random(42) # Seeded for consistency if this code path is hit
        colors = ['#%06X' % rng.randint(0, 0xFFFFFF) for i in range(optimal_k)]


    for i in range(optimal_k):
        cluster_data = df_pca[df_pca['Cluster'] == i]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'],
                    label=f'Cluster {i}', color=colors[i % len(colors)], alpha=0.7, s=50)

    plt.title("PCA of Beer Clusters (Validation Set)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True)

    # Add analysis text to the plot (optional, can also be in comments)
    # Silhouette score for k=2 on validation was ~0.4391
    # Davies-Bouldin for k=2 on validation was ~0.8834
    analysis_text = (
        f"PCA visualization of K-Means (k={optimal_k}) clusters.\n"
        "The plot shows the distribution of clusters in the 2D space defined by the first two principal components.\n"
        "Observations:\n"
        "- The clusters show some visual separation, though with potential overlap.\n"
        "- This level of separation is consistent with a Silhouette Score of ~0.44 (validation set).\n"
        "  Scores closer to 1 would imply denser, more well-separated clusters.\n"
        "- The Davies-Bouldin Index of ~0.88 (validation set) also suggests that clusters are not perfectly distinct;\n"
        "  lower values (closer to 0) are better for DB index."
    )
    # plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})


    pca_plot_path = os.path.join(output_dir, "pca_cluster_visualization.png")
    try:
        plt.savefig(pca_plot_path) #, bbox_inches='tight' if using figtext and want to ensure it's included
        print(f"\nPCA cluster visualization saved to '{pca_plot_path}'")
    except Exception as e:
        print(f"Error saving PCA plot: {e}")
    plt.close()

    # --- 5. Analysis (in script comments as requested) ---
    #
    # Visual Separation:
    # The PCA plot visually represents the clusters in a 2D space. For k=2, we typically look for two
    # distinct groups of points. The degree of separation (or overlap) gives a qualitative sense
    # of cluster quality.
    #
    # Relation to Metrics (Validation Set k=2):
    # - Silhouette Score: ~0.4391. A score in this range (0.25 to 0.5) suggests reasonable structure
    #   but that clusters are not very dense or well-separated. The PCA plot is expected to show
    #   this: distinct groupings but likely not perfectly spherical, and potentially some points
    #   close to the boundary between clusters or some degree of overlap.
    # - Davies-Bouldin Index: ~0.8834. Values closer to 0 are better. A score around 0.88
    #   also indicates that the clusters are not perfectly compact and far from each other.
    #   The visual representation in PCA should align with this, showing clusters that might be
    #   somewhat diffuse or not entirely separated.
    #
    # Conclusion from PCA Plot:
    # If the PCA plot shows two relatively distinct clouds of points, it supports the choice of k=2.
    # If the clouds heavily overlap or are not distinct, it might suggest that either:
    #   a) k=2 is not ideal (though metrics suggested it was the best among options).
    #   b) The clusters are inherently not well-separable in a linear fashion or via PCA's projection.
    #   c) The chosen features, while scaled, might not perfectly lend themselves to strong clustering.
    # The primary purpose here is to see if the k=2 structure found by K-means is somewhat discernible
    # in a lower-dimensional view.
    #
    # Based on the metrics, we expect visible groupings but not perfectly clean separation.

if __name__ == "__main__":
    visualize_pca_clusters()
