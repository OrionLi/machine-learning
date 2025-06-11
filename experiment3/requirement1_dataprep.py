import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np # For stratify if y is not pandas Series initially

def prepare_creditcard_data(output_dir="experiment3"):
    """
    Loads credit card fraud data, preprocesses it, and splits into
    train, validation, and test sets. Saves them to CSV files.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- 1. Define dataset path and check for existence ---
    dataset_path = 'resources/creditcard.csv'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at '{dataset_path}'.")
        print("Please download it from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
        print("Then place 'creditcard.csv' in the 'resources' directory relative to the project root.")
        exit() # Exit the script as per requirement

    # --- 2. Load Data (if exists) ---
    try:
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded '{dataset_path}'. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        exit()

    # --- 3. Feature Selection & Preprocessing ---
    # Scale 'Amount'
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

    # Scale 'Time'
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop original 'Time' and 'Amount' columns
    df = df.drop(['Time', 'Amount'], axis=1)
    print("Scaled 'Amount' and 'Time' columns, and dropped originals.")

    # Define features X and target y
    X = df.drop('Class', axis=1)
    y = df['Class']
    print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    # --- 4. Dataset Splitting ---
    # Stratify is important for imbalanced datasets like fraud detection.

    # Split into temporary training (85%) and testing (15%)
    # (original_total * 0.85 for temp, original_total * 0.15 for test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )
    print(f"Split into temporary training set ({X_temp.shape[0]} samples) and test set ({X_test.shape[0]} samples).")

    # Split temporary training into actual training (70% of original) and validation (15% of original)
    # The temp set is 85% of original. We need 15% of original for validation from this 85%.
    # So, validation_size_relative_to_temp = (15% of original) / (85% of original) = 15/85.
    validation_relative_size = 0.15 / 0.85

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_relative_size,
        random_state=42,
        stratify=y_temp
    )
    print(f"Split temporary training into actual training set ({X_train.shape[0]} samples) and validation set ({X_val.shape[0]} samples).")

    # Verify final distribution (approximate due to integer sample counts)
    print(f"Final distribution: Train ~{X_train.shape[0]/df.shape[0]*100:.0f}%, Validation ~{X_val.shape[0]/df.shape[0]*100:.0f}%, Test ~{X_test.shape[0]/df.shape[0]*100:.0f}%")


    # --- 5. Save Data ---
    try:
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = os.path.join(output_dir, "train_data.csv")
        val_path = os.path.join(output_dir, "validation_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nData prepared and saved to:")
        print(f"  Training data: '{train_path}' ({train_df.shape[0]} samples)")
        print(f"  Validation data: '{val_path}' ({val_df.shape[0]} samples)")
        print(f"  Test data: '{test_path}' ({test_df.shape[0]} samples)")

    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    prepare_creditcard_data()
