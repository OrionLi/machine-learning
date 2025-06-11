import pandas as pd
import numpy as np
import os
# Attempt to import PyTorch and related libraries
# These will only be available if the installation was successful
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for non-interactive environments
    import matplotlib.pyplot as plt
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch or other required libraries (sklearn, matplotlib) not fully available. Please check installation.")
    # Define dummy classes/functions if needed for the script structure to be parseable,
    # though the script will exit early if data files are missing or PyTorch is needed.

if PYTORCH_AVAILABLE:
    class LogisticRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LogisticRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

def run_pytorch_logistic_regression(output_dir="experiment3"):
    """
    Loads preprocessed credit card data, trains a PyTorch Logistic Regression model,
    visualizes loss, and evaluates on a validation set.
    """
    if not PYTORCH_AVAILABLE:
        print("Exiting script as PyTorch is not available.")
        exit() # Exit if PyTorch couldn't be imported.

    # --- 1. Define file paths and check for data ---
    train_file_path = os.path.join(output_dir, 'train_data.csv')
    val_file_path = os.path.join(output_dir, 'validation_data.csv')

    if not os.path.exists(train_file_path) or not os.path.exists(val_file_path):
        print("Training/validation data not found in 'experiment3/' directory.")
        print("Please ensure 'creditcard.csv' was downloaded to 'resources/' and")
        print("'experiment3/requirement1_dataprep.py' ran successfully to generate these files.")
        exit()

    # --- 2. Load Data ---
    try:
        train_df = pd.read_csv(train_file_path)
        val_df = pd.read_csv(val_file_path)
        print(f"Loaded training data: {train_df.shape}, Validation data: {val_df.shape}")
    except Exception as e:
        print(f"Error loading data CSVs: {e}")
        exit()

    # Separate features (X) and target (y)
    X_train_pd = train_df.drop('Class', axis=1)
    y_train_pd = train_df['Class']
    X_val_pd = val_df.drop('Class', axis=1)
    y_val_pd = val_df['Class']

    # --- 3. Convert to PyTorch Tensors ---
    X_train_tensor = torch.tensor(X_train_pd.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_pd.values, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_pd.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_pd.values, dtype=torch.float32).unsqueeze(1)
    print("Data converted to PyTorch tensors.")

    # --- 4. Instantiate Model, Loss, Optimizer ---
    input_dim = X_train_tensor.shape[1]
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model, Loss function, and Optimizer initialized.")

    # --- 5. Training Loop ---
    epochs = 100  # Can be adjusted
    train_losses = []
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    # --- 6. Visualize Loss ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (BCELoss)')
        plt.title('Training Loss vs. Epochs')
        plt.legend()
        loss_plot_path = os.path.join(output_dir, "training_loss.png")
        plt.savefig(loss_plot_path)
        print(f"Training loss plot saved to '{loss_plot_path}'")
        plt.close()
    except Exception as e:
        print(f"Error saving loss plot: {e}")


    # --- 7. Evaluation on Validation Set ---
    model.eval()  # Set model to evaluation mode
    print("\nEvaluating on validation set...")
    with torch.no_grad():  # Disable gradient calculation for evaluation
        val_outputs = model(X_val_tensor)
        # Apply threshold (0.5 for sigmoid) to get binary predictions
        predicted = (val_outputs >= 0.5).float()

        # Calculate metrics
        accuracy = accuracy_score(y_val_tensor.numpy(), predicted.numpy())
        # For precision, recall, f1, ensure positive label is 1, handle zero division
        precision = precision_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        recall = recall_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        f1 = f1_score(y_val_tensor.numpy(), predicted.numpy(), pos_label=1, zero_division=0)
        conf_matrix = confusion_matrix(y_val_tensor.numpy(), predicted.numpy())

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision (for fraud class 1): {precision:.4f}')
    print(f'Validation Recall (for fraud class 1): {recall:.4f}')
    print(f'Validation F1 Score (for fraud class 1): {f1:.4f}')
    print(f'Validation Confusion Matrix:\n{conf_matrix}')

if __name__ == "__main__":
    run_pytorch_logistic_regression()
