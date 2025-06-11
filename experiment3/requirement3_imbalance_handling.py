import pandas as pd
import numpy as np
import os

# --- Check for Core Dependencies ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Please check installation.")

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn not available. Please check installation.")

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive environments
    import matplotlib.pyplot as plt # Though not used in this version, good for consistency
    SKLEARN_MATPLOTLIB_AVAILABLE = True
except ImportError:
    SKLEARN_MATPLOTLIB_AVAILABLE = False
    print("scikit-learn or matplotlib not fully available. Please check installation.")


# --- Define Model (only if PyTorch is available) ---
if PYTORCH_AVAILABLE:
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

# --- Helper function to train and evaluate ---
def train_and_evaluate(model, X_train_np, y_train_np, X_val_tensor, y_val_tensor, description):
    if not PYTORCH_AVAILABLE or not SKLEARN_MATPLOTLIB_AVAILABLE:
        print(f"Skipping train_and_evaluate for '{description}' due to missing libraries.")
        return

    print(f"\n--- Results for {description} ---")

    # Convert training data to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)

    # Instantiate optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    epochs = 50 # Reduced epochs for faster runs with resampled data, can be adjusted
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        predicted = (val_outputs >= 0.5).float()

        accuracy = accuracy_score(y_val_tensor.numpy(), predicted.numpy())
        precision = precision_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        recall = recall_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        f1 = f1_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        conf_matrix = confusion_matrix(y_val_tensor.numpy(), predicted.numpy())

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision (for fraud class 1): {precision:.4f}')
    print(f'Validation Recall (for fraud class 1): {recall:.4f}')
    print(f'Validation F1 Score (for fraud class 1): {f1:.4f}')
    print(f'Validation Confusion Matrix:\n{conf_matrix}')


def apply_imbalance_techniques(output_dir="experiment3"):
    """
    Loads data, applies undersampling and oversampling, trains models,
    and evaluates them.
    """
    if not PYTORCH_AVAILABLE or not IMBLEARN_AVAILABLE or not SKLEARN_MATPLOTLIB_AVAILABLE:
        print("Exiting script due to missing core libraries (PyTorch, imbalanced-learn, scikit-learn, or matplotlib).")
        exit()

    # --- Define file paths and check for data ---
    train_file_path = os.path.join(output_dir, 'train_data.csv')
    val_file_path = os.path.join(output_dir, 'validation_data.csv')

    if not os.path.exists(train_file_path) or not os.path.exists(val_file_path):
        print("\nTraining/validation data not found in 'experiment3/' directory.")
        print("Please ensure 'creditcard.csv' was downloaded to 'resources/' and")
        print("'experiment3/requirement1_dataprep.py' ran successfully to generate these files.")
        exit()

    # --- Load Data ---
    try:
        train_df = pd.read_csv(train_file_path)
        val_df = pd.read_csv(val_file_path)
        print(f"Loaded training data: {train_df.shape}, Validation data: {val_df.shape}")
    except Exception as e:
        print(f"Error loading data CSVs: {e}")
        exit()

    X_train_original_pd = train_df.drop('Class', axis=1)
    y_train_original_pd = train_df['Class']
    X_val_pd = val_df.drop('Class', axis=1)
    y_val_pd = val_df['Class']

    # Convert validation data to PyTorch Tensors (once)
    X_val_tensor = torch.tensor(X_val_pd.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_pd.values, dtype=torch.float32).unsqueeze(1)

    input_dim_original = X_train_original_pd.shape[1] # Should be same for all X

    # --- Undersampling ---
    print("\n--- Applying Random Undersampling ---")
    rus = RandomUnderSampler(random_state=42)
    X_train_under_np, y_train_under_np = rus.fit_resample(X_train_original_pd, y_train_original_pd)
    # Convert to numpy explicitly if not already (fit_resample can return pandas series/df)
    if isinstance(X_train_under_np, pd.DataFrame): X_train_under_np = X_train_under_np.values
    if isinstance(y_train_under_np, pd.Series): y_train_under_np = y_train_under_np.values

    print(f"Shape after undersampling: X_train_under: {X_train_under_np.shape}, y_train_under: {y_train_under_np.shape}")
    model_under = LogisticRegressionModel(input_dim_original)
    train_and_evaluate(model_under, X_train_under_np, y_train_under_np, X_val_tensor, y_val_tensor, "Undersampled Training Data")
    print("\nDiscussion for Undersampling:")
    print(" - Random Undersampling balances the dataset by removing samples from the majority class.")
    print(" - Pros: Can significantly speed up training, may help model focus on minority class signals.")
    print(" - Cons: Loss of potentially useful information from the majority class, can lead to poorer generalization if removed data was important.")
    print(" - Expected: Often higher recall for the minority (fraud) class, but potentially lower overall accuracy and precision on the original distribution if the model becomes too biased towards the resampled view.")

    # --- Oversampling (RandomOverSampler) ---
    print("\n--- Applying Random Oversampling ---")
    ros = RandomOverSampler(random_state=42)
    X_train_over_np, y_train_over_np = ros.fit_resample(X_train_original_pd, y_train_original_pd)
    if isinstance(X_train_over_np, pd.DataFrame): X_train_over_np = X_train_over_np.values
    if isinstance(y_train_over_np, pd.Series): y_train_over_np = y_train_over_np.values

    print(f"Shape after oversampling: X_train_over: {X_train_over_np.shape}, y_train_over: {y_train_over_np.shape}")
    model_over = LogisticRegressionModel(input_dim_original)
    train_and_evaluate(model_over, X_train_over_np, y_train_over_np, X_val_tensor, y_val_tensor, "Oversampled Training Data (Random)")
    print("\nDiscussion for Random Oversampling:")
    print(" - Random Oversampling balances by duplicating samples from the minority class.")
    print(" - Pros: No loss of information from majority class, can improve sensitivity to minority class.")
    print(" - Cons: Can lead to overfitting as the model sees exact copies of minority samples. Does not generate new synthetic data like SMOTE.")
    print(" - Expected: May improve recall for the minority class. Risk of overfitting if not validated carefully. Overall performance can vary.")

    print("\n--- Comparison Note ---")
    print("Compare these results (especially Precision, Recall, F1 for class 1) with the baseline model")
    print("from 'requirement2_pytorch_logistic_regression.py' (if it ran with the original imbalanced data)")
    print("to understand the impact of these imbalance handling techniques on model performance for the fraud class.")

if __name__ == "__main__":
    apply_imbalance_techniques()
