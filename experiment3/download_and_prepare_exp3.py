import os
import pandas as pd
import requests # For attempting download
import zipfile # For unzipping if download were real
import io # For handling bytes stream from download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys # For sys.exit()

# --- 1. Define Constants ---
KAGGLE_DATASET_URL = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download'
RESOURCES_DIR = 'resources'
TARGET_CSV = 'creditcard.csv'
# Kaggle often zips datasets, even if it's a single CSV.
# The zip file name on Kaggle for this dataset is 'creditcardfraud.zip'.
# Inside this zip is 'creditcard.csv'.
ZIP_FILENAME_KAGGLE = 'creditcardfraud.zip'

EXPECTED_CSV_PATH = os.path.join(RESOURCES_DIR, TARGET_CSV)

OUTPUT_DIR_EXP3 = 'experiment3'
TRAIN_CSV = os.path.join(OUTPUT_DIR_EXP3, 'train_data.csv')
VAL_CSV = os.path.join(OUTPUT_DIR_EXP3, 'validation_data.csv')
TEST_CSV = os.path.join(OUTPUT_DIR_EXP3, 'test_data.csv')

def download_and_prepare():
    """
    Main function to attempt download, then load, process, and split data.
    """
    # --- 2. Ensure directories exist ---
    os.makedirs(RESOURCES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR_EXP3, exist_ok=True)
    print(f"Ensured directories '{RESOURCES_DIR}' and '{OUTPUT_DIR_EXP3}' exist.")

    # --- 3. Attempt to obtain creditcard.csv ---
    if os.path.exists(EXPECTED_CSV_PATH):
        print(f"'{EXPECTED_CSV_PATH}' already exists. Skipping download.")
    else:
        print(f"\n'{EXPECTED_CSV_PATH}' not found. Attempting to handle download/extraction...")
        # Note: Direct download from Kaggle dataset pages is generally not feasible
        # without using the Kaggle API or being logged into a session.
        # The URL provided is a landing page, not a direct file link.
        # This section simulates the difficulty and guides manual download.

        # Path where a downloaded zip might be temporarily stored
        temp_zip_path = os.path.join(RESOURCES_DIR, ZIP_FILENAME_KAGGLE)

        print(f"Attempting simulated download from Kaggle: {KAGGLE_DATASET_URL}")
        print("Note: Direct HTTP GET requests to Kaggle dataset pages usually fetch HTML or require authentication.")
        print("This 'download attempt' is illustrative and will simulate failure,")
        print("as reliable direct download links are not typically available without the Kaggle API or manual browser interaction.")

        try:
            # This is where actual Kaggle API download logic would go, or a more complex web scraping
            # that handles login and dynamic link generation (which is fragile and not recommended).
            # For this exercise, we simulate the failure of a naive direct download.
            # A real requests.get(KAGGLE_DATASET_URL) would get HTML.
            # A real direct file link (if one could be found and was stable) would be used here.

            # Simulate that a direct download attempt for the ZIP file failed.
            raise Exception("Simulated Kaggle download failure. Direct public links to dataset ZIPs are unreliable or require API/login.")

            # If a hypothetical direct download of the ZIP were successful:
            # print(f"Simulated: Downloaded '{ZIP_FILENAME_KAGGLE}' to '{temp_zip_path}'.")
            # print(f"Simulated: Extracting '{TARGET_CSV}' from '{temp_zip_path}'...")
            # with zipfile.ZipFile(temp_zip_path, 'r') as z:
            #     z.extract(TARGET_CSV, RESOURCES_DIR)
            # print(f"Simulated: Successfully extracted '{TARGET_CSV}' to '{RESOURCES_DIR}'.")
            # if os.path.exists(temp_zip_path):
            #     os.remove(temp_zip_path) # Clean up temp zip

        except Exception as e:
            print(f"\nDownload attempt failed: {e}")
            print(f"-------------------------------------------------------------------------------------------")
            print(f"**ACTION REQUIRED**:")
            print(f"Please manually download the dataset 'creditcardfraud.zip' from Kaggle:")
            print(f"  {KAGGLE_DATASET_URL}")
            print(f"Then, extract 'creditcard.csv' from the zip file.")
            print(f"Place the extracted 'creditcard.csv' into the '{RESOURCES_DIR}' directory.")
            print(f"Expected final path: '{EXPECTED_CSV_PATH}'")
            print(f"-------------------------------------------------------------------------------------------")


    # --- 4. Load and Process Data (if CSV is available) ---
    if not os.path.exists(EXPECTED_CSV_PATH):
        print(f"\nError: '{EXPECTED_CSV_PATH}' not found after download/check attempts.")
        print("Please ensure the file is correctly placed and try again.")
        sys.exit(1) # Exit script if file is still not there

    print(f"\nLoading data from '{EXPECTED_CSV_PATH}'...")
    try:
        df = pd.read_csv(EXPECTED_CSV_PATH)
        print(f"Successfully loaded '{EXPECTED_CSV_PATH}'. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV file '{EXPECTED_CSV_PATH}': {e}")
        sys.exit(1)

    # Preprocessing steps (same as requirement1_dataprep.py)
    try:
        print("Preprocessing data (scaling 'Amount' and 'Time', dropping originals)...")
        scaler = StandardScaler()
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df_processed = df.drop(['Time', 'Amount'], axis=1)

        X = df_processed.drop('Class', axis=1)
        y = df_processed['Class']
        print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

        # Dataset Splitting
        print("Splitting data into training, validation, and test sets (70/15/15)...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        validation_relative_size = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_relative_size, random_state=42, stratify=y_temp
        )

        # Save Data
        print("Saving processed datasets...")
        train_df_out = pd.concat([X_train, y_train], axis=1)
        val_df_out = pd.concat([X_val, y_val], axis=1)
        test_df_out = pd.concat([X_test, y_test], axis=1)

        train_df_out.to_csv(TRAIN_CSV, index=False)
        val_df_out.to_csv(VAL_CSV, index=False)
        test_df_out.to_csv(TEST_CSV, index=False)

        print(f"\nData preparation complete. Processed files saved:")
        print(f"  Training data: '{TRAIN_CSV}' ({train_df_out.shape[0]} samples)")
        print(f"  Validation data: '{VAL_CSV}' ({val_df_out.shape[0]} samples)")
        print(f"  Test data: '{TEST_CSV}' ({test_df_out.shape[0]} samples)")
        print(f"Approximate final distribution: Train ~{X_train.shape[0]/df.shape[0]*100:.0f}%, Validation ~{X_val.shape[0]/df.shape[0]*100:.0f}%, Test ~{X_test.shape[0]/df.shape[0]*100:.0f}%")

    except Exception as e:
        print(f"Error during data processing or saving: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_and_prepare()
