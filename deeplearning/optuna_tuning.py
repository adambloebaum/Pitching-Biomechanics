import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import os

# Load the scaled datasets
X_train_scaled = pd.read_csv('X_train_scaled.csv').values
X_val_scaled = pd.read_csv('X_val_scaled.csv').values
X_test_scaled = pd.read_csv('X_test_scaled.csv').values

# Load the labels
y_train = pd.read_csv('y_train.csv').values.ravel()
y_val = pd.read_csv('y_val.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initialize model
class PitchVelocityModel(nn.Module):
    def __init__(self, num_features, layer_sizes, dropout_prob):
        super(PitchVelocityModel, self).__init__()
        
        # Assuming layer_sizes is a list with sizes for fc1, fc2, fc3, fc4, fc5, fc6 respectively
        self.fc1 = nn.Linear(num_features, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc5 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.fc6 = nn.Linear(layer_sizes[4], layer_sizes[5])
        self.fc7 = nn.Linear(layer_sizes[5], 1)

        # Dropout function, only for fc2 and fc4
        self.dropout = nn.Dropout(dropout_prob)

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))  # Dropout after fc2
        x = self.relu(self.fc3(x))
        x = self.dropout(self.relu(self.fc4(x)))  # Dropout after fc4
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)

        return x
    
# Objective function for Optuna
def objective(trial):
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [64, 32])
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)

    # Define the layer sizes (you can also optimize these if needed)
    layer_sizes = [
        trial.suggest_int('fc1_units', 512, 1024),
        trial.suggest_int('fc2_units', 32, 1024),
        trial.suggest_int('fc3_units', 32, 1024),
        trial.suggest_int('fc4_units', 32, 1024),
        trial.suggest_int('fc5_units', 32, 1024),
        trial.suggest_int('fc6_units', 32, 128)
    ]

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model with the suggested layer sizes and dropout probability
    model = PitchVelocityModel(X_train_tensor.shape[1], layer_sizes, dropout_prob)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    epochs=100

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Report intermediate objective value to Optuna
        trial.report(loss.item(), epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

        scheduler.step()

    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred = y_pred.squeeze()
            val_losses.append(criterion(y_pred, y_batch).item())

    # Calculate mean squared error
    mean_val_loss = np.mean(val_losses)

    return mean_val_loss

# Creating the Optuna study object
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500, n_jobs=-1)

# Best hyperparameters
best_params = study.best_params
print('Best trial:', best_params)

# Convert the dictionary of best parameters to a string
best_params_str = "\n".join(f"{key}: {value}" for key, value in best_params.items())

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path for the output file
output_file_path = os.path.join(script_dir, 'best_params.txt')

# Write to a text file
with open(output_file_path, "w") as file:
    file.write(best_params_str)
