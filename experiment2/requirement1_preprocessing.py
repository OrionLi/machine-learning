import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_beer_data(output_dir="experiment2"):
    """
    Loads beer data, selects features, cleans, scales, and splits into train/validation sets.
    Saves processed sets to CSV files.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- 1. Load Data ---
    try:
        df_ratings = pd.read_csv("resources/Lab02-beer_profile_and_ratings.csv")
        df_descriptors = pd.read_excel("resources/Lab02-Beer Descriptors Simplified.xlsx")
        print("Successfully loaded 'beer_profile_and_ratings.csv' and 'Lab02-Beer Descriptors Simplified.xlsx'")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure files are in the 'resources' directory.")
        return

    # --- 2. Initial Exploration and Merging Decision ---
    print("\nColumns in beer_profile_and_ratings.csv:")
    print(df_ratings.columns.tolist())
    print(f"Shape: {df_ratings.shape}")

    print("\nColumns in Lab02-Beer Descriptors Simplified.xlsx:")
    print(df_descriptors.columns.tolist())
    print(f"Shape: {df_descriptors.shape}")

    # Decision: Prioritize df_ratings. Merging df_descriptors is complex due to name variations
    # and the presence of fuzzy matching lists (implying direct merge is insufficient).
    # This script will proceed using only df_ratings.
    print("\nProceeding with 'beer_profile_and_ratings.csv' as the primary dataset for feature extraction.")
    df_main = df_ratings.copy()

    # --- 3. Feature Selection ---
    # Selected features: review scores and beer ABV.
    # These are numeric and relevant for clustering based on beer characteristics and perception.
    feature_columns = [
        'review_overall', 'review_aroma', 'review_appearance',
        'review_palate', 'review_taste', 'ABV' # Corrected from 'beer_abv' to 'ABV'
    ]
    print(f"\nSelected features: {feature_columns}")

    # Check if selected features exist
    missing_features = [col for col in feature_columns if col not in df_main.columns]
    if missing_features:
        print(f"Error: The following selected features are missing from the dataset: {missing_features}")
        return

    df_processed = df_main[feature_columns].copy()

    # --- 4. Data Cleaning ---
    print(f"\nInitial shape of data with selected features: {df_processed.shape}")
    print(f"Initial missing values:\n{df_processed.isnull().sum()}")

    # Convert to numeric, coercing errors (e.g., if 'ABV' has non-numeric entries)
    for col in feature_columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Handle missing values created by coercion or already present
    # For 'ABV', special attention might be needed if NaNs are extensive after coercion
    # Note: The original column name in df_main for ABV is 'ABV'.
    # The df_processed Series for ABV is df_processed['ABV'].
    if 'ABV' in df_processed.columns and df_processed['ABV'].isnull().any():
        # Example: if 'Not Available' was coerced to NaN for ABV.
        # We'll impute with mean, but one could also consider median or removing rows if critical.
        # df_main['ABV'] would be the original Series to check for initial NaNs vs post-coercion.
        print(f"Coerced non-numeric ABVs to NaN. Original ABV NaNs in df_main: {df_main['ABV'].isnull().sum()}, Post-coercion ABV NaNs in df_processed: {df_processed['ABV'].isnull().sum()}")

    # Impute NaNs with mean for all selected numeric features
    for col in feature_columns:
        if df_processed[col].isnull().any():
            mean_val = df_processed[col].mean()
            df_processed[col].fillna(mean_val, inplace=True)
            print(f"Imputed NaNs in '{col}' with mean: {mean_val:.2f}")

    print(f"\nMissing values after imputation:\n{df_processed.isnull().sum()}")

    # Outlier handling for ABV (simple example: cap at 20%)
    # A more robust method would use IQR, but this is a quick check.
    if 'ABV' in feature_columns: # Check with corrected column name
        original_abv_max = df_processed['ABV'].max()
        cap_abv = 20.0
        df_processed['ABV'] = df_processed['ABV'].apply(lambda x: min(x, cap_abv) if pd.notnull(x) else x)
        if original_abv_max > cap_abv:
            print(f"Capped 'ABV' at {cap_abv}%. Original max was {original_abv_max:.2f}%.")
        # Also handle potentially very low or zero ABVs if they are errors
        # Ensure we are not calculating mean from a series that might include zeros or negatives if that's not desired
        mean_positive_abv = df_processed.loc[df_processed['ABV'] > 0, 'ABV'].mean()
        if pd.notnull(mean_positive_abv): # Check if mean_positive_abv is not NaN (e.g. if all ABVs were <=0)
             df_processed.loc[df_processed['ABV'] <= 0, 'ABV'] = mean_positive_abv
        else: # Fallback if all ABVs were <=0 or NaN initially, replace 0 or negative ABVs with a small positive default or handle as error
             df_processed.loc[df_processed['ABV'] <= 0, 'ABV'] = 0.05 # Or np.nan to be dropped/re-imputed
             print("Warning: Could not calculate mean of positive ABVs to impute zero/negative ABVs. Used default or NaN.")


    # Drop rows if any critical feature is still NaN (should not happen with mean imputation, but as a safeguard)
    df_processed.dropna(inplace=True)
    print(f"Shape after NaN handling: {df_processed.shape}")

    if df_processed.empty:
        print("Error: No data remaining after cleaning. Check input data and cleaning steps.")
        return

    # --- 5. Data Transformation (Scaling) ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed)
    df_scaled = pd.DataFrame(scaled_features, columns=feature_columns, index=df_processed.index)
    print("\nData scaled using StandardScaler.")

    # --- 6. Dataset Splitting ---
    train_df, val_df = train_test_split(
        df_scaled,
        test_size=0.2,
        random_state=42  # For reproducibility
    )
    print(f"\nData split into training ({train_df.shape[0]} samples) and validation ({val_df.shape[0]} samples) sets.")

    # --- 7. Save Processed Data ---
    try:
        train_path = os.path.join(output_dir, "train_data.csv")
        val_path = os.path.join(output_dir, "validation_data.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        print(f"Processed training data saved to '{train_path}'")
        print(f"Processed validation data saved to '{val_path}'")
    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    preprocess_beer_data()
