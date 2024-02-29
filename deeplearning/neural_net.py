import pandas as pd
import numpy as np
import mysql.connector
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os

print("Loading data...")

# Configure and connect to database
db_config = {
    "host": "HOST",
    "user": "USER",
    "password": "PASSWORD",
    "database": "DATABASE"
}

connection = mysql.connector.connect(**db_config)
query = 'SELECT * FROM table INNER JOIN table ON table.session = table.session;'
df = pd.read_sql(query, connection)
connection.close()

# Drop columns with nan values
columns_with_nan = df.columns[df.isna().any()].tolist()
cleaned_df = df.drop(columns=columns_with_nan)

# Drop unneeded columns
columns_to_exclude = ['session_pitch', 'session', 'p_throws', 'pitch_type', 'session', 'session_date', 'user', 'session_tag', 'session_location', 'playing_level', 'irb']
cleaned_df = cleaned_df.drop(columns=columns_to_exclude)

# Engineer features relative to body weight
bw = 'session_mass_kg'
exclude = (bw, 'pitch_speed_mph')
for col in cleaned_df.columns:
    if col not in exclude:
        cleaned_df[f'{col}_rel_{bw}'] = cleaned_df[col] / cleaned_df[bw]

#  Split features and Labels
X = cleaned_df.drop('pitch_speed_mph', axis=1)
y = cleaned_df['pitch_speed_mph']

# Split the data into 80% train and 20% combined validation and test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the temporary data into 50% validation and 50% test (10% of total each)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both training, validation, and test data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaled data
pd.DataFrame(X_train_scaled).to_csv('X_train_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
pd.DataFrame(X_val_scaled).to_csv('X_val_scaled.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('X_test_scaled.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create TensorDatasets
train_data = TensorDataset(X_train_tensor, y_train_tensor)
val_data = TensorDataset(X_val_tensor, y_val_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class PitchVelocityModel(nn.Module):
    def __init__(self, num_features):
        super(PitchVelocityModel, self).__init__()
        # Fully connected layers
        self.fc1 = torch.nn.Linear(num_features, 561)
        self.fc2 = torch.nn.Linear(561, 526)
        self.fc3 = torch.nn.Linear(526, 33)
        self.fc4 = torch.nn.Linear(33, 545)
        self.fc5 = torch.nn.Linear(545, 808)
        self.fc6 = torch.nn.Linear(808, 97)
        self.fc7 = torch.nn.Linear(97, 1)

        # Dropout function
        self.dropout = nn.Dropout(0.13113473600982906)
        
        # Relu activation function
        self.relu = torch.nn.ReLU()  
                
    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        
        return x
    
model = PitchVelocityModel(X_train.shape[1])

# Check for GPU availability and enable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Mean Squared Error Loss for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005457544627826253)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# List to store losses
train_losses = []
val_losses = []

# Early stopping parameters
best_val_loss = float('inf')
patience, trials = 10, 0

epochs = 200
print("Beginning training...")
for epoch in range(epochs + 1):
    # Training phase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_loss = criterion(y_pred.squeeze(), y_batch)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        trials = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    print(f'Epoch {epoch}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Plotting the training and validation losses
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, 'outputs')
plot_path = os.path.join(outputs_dir, 'loss_plot.png')

plt.savefig(plot_path)

# Evaluate on Test Set
model.eval()
y_pred_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())

y_pred_flat = np.concatenate(y_pred_list).ravel()
y_test_numpy = y_test_tensor.cpu().numpy().ravel()
mse = mean_squared_error(y_test_numpy, y_pred_flat)
print(f'Test Mean Squared Error: {mse}')

# Paths for saving the scaler and the model
scaler_path = os.path.join(script_dir, 'scaler.pk1')
model_path = os.path.join(script_dir, 'pitch_velocity_model.pth')

# Save the scaler and the model
joblib.dump(scaler, scaler_path)
torch.save(model.state_dict(), model_path)
