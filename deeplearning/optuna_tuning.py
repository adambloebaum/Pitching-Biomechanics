import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import deeplearning.optuna_tuning as optuna_tuning

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

# Initialize model
class PitchVelocityModel(nn.Module):
    def __init__(self, num_features):
        super(PitchVelocityModel, self).__init__()
        # Fully connected layers
        self.fc1 = torch.nn.Linear(num_features, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 32)
        self.fc7 = torch.nn.Linear(32, 1)
        
        # Relu activation function
        self.relu = torch.nn.ReLU()  
                
    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        
        return x

# Objective function for Optuna
def objective(trial):
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [64, 32, 24])
    lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model setup
    model = PitchVelocityModel(X_train_scaled.shape[1])
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs=100

    # Training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()

    return val_loss / len(val_loader)

# Creating the Optuna study object
study = optuna_tuning.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best hyperparameters
best_params = study.best_params
print('Best trial:', best_params)
