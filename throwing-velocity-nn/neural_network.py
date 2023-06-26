# Import necessary libraries
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load in data
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)

# Preprocess to remove empty or object type values
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.drop('session', axis=1)
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])

# Split the data into features and targets
features = poi_metrics.drop('pitch_speed_mph', axis=1)
targets = poi_metrics['pitch_speed_mph']

# Create feature names to be used later
feature_names = features.columns

# Train test split for features and targets
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.1, random_state=42)

# Create unscaled test features for later analysis
unscaled_test_features = test_features

# Standard scaling the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)

test_features_np = test_features

# Define model
class Net(torch.nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(train_features.shape[1], 1024)
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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        
        return x

model = Net()

# Define hyperparameters
learning_rate = 0.0035
epochs = 100
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model

# Identify tracked values
train_loss_list = np.zeros((epochs,))

# DataFrames are converted to tensors
train_features = torch.from_numpy(train_features).float()
train_targets = torch.from_numpy(train_targets.values).float()  # changed to float
test_features = torch.from_numpy(test_features).float()
test_targets = torch.from_numpy(test_targets.values).float()  # changed to float

# Training loop
for epoch in tqdm.trange(epochs):
    
    optimizer.zero_grad()
    
    y_pred = model(train_features)
    loss = loss_func(y_pred, train_targets)
    train_loss_list[epoch] = loss.item()
    
    loss.backward()
    optimizer.step()
    
    # Print loss per epoch
    print(f"Epoch {epoch}, MSE: {loss.item()}")

blue = (169/255, 196/255, 235/255, 1)

plt.rcParams.update({'font.size': 20})

# Plot training loss over epochs
plt.figure(figsize = (12, 6))
plt.plot(train_loss_list, linewidth = 3, color=blue)
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.title("Training Loss Over Epochs")
sns.despine()

# Set model to eval mode
model.eval()

# Predict targets with test features
with torch.no_grad():
    y_test = model(test_features)
    
# Print the predicted and the actual values from the test data
for i in range(len(y_test)):
    print(f"Prediction: {y_test[i]}, Actual: {test_targets[i]}")
    
# Convert tensors back to numpy arrays
y_test = y_test.detach().numpy()
test_targets = test_targets.detach().numpy()

# Compute and print the average difference between predicted and actual values from the test data
diff = np.abs(y_test-test_targets)
avg_diff = np.mean(diff)
print(f"Average difference: {np.round(avg_diff, 2)}mph")

# Enable gradient calculation for the input tensor
test_features.requires_grad_(True)

# Forward pass to calculate the output
output = model(test_features)

# Calculate gradients of the output with respect to the input
output.backward(torch.ones_like(output))

# Get the gradients of the input tensor
gradients = test_features.grad

# Calculate the feature importance based on the absolute gradients
abs_gradients = torch.abs(gradients)
avg_gradients = torch.mean(abs_gradients, dim=0)
feature_importance = avg_gradients.detach().numpy()

# Sort features from most to least importance
sorted_indices = np.argsort(feature_importance)
sorted_feature_importance = feature_importance[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Plot feature importance
plt.figure(figsize=(10, 10))
plt.tight_layout()
plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance)
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names, fontsize=8)
plt.xticks([])
plt.show()

# Define the feature index for shoulder horizontal abduction at foot plant
feature_index = 9

# Extract the feature from the test features
feature_values = test_features[:, feature_index].detach().numpy()

# Create an empty array to store the predicted outputs
predicted_outputs = np.zeros_like(feature_values)

# Compute the predictions for different feature values
for i, value in enumerate(feature_values):
    test_features_copy = test_features.clone()
    test_features_copy[:, feature_index] = torch.tensor(value)
    outputs = model(test_features_copy)
    predicted_outputs[i] = outputs.mean().item()  # Store the mean prediction
    
# Fit a polynomial regression model to the data
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(feature_values.reshape(-1, 1))
regressor = LinearRegression()
regressor.fit(X_poly, predicted_outputs)

# Predict outputs for the range of feature values
feature_values_range = np.linspace(np.min(feature_values), np.max(feature_values), 100)
X_poly_range = poly_features.transform(feature_values_range.reshape(-1, 1))
predicted_outputs_range = regressor.predict(X_poly_range)

# Plot the partial dependence and trendline
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, predicted_outputs)
plt.plot(feature_values_range, predicted_outputs_range, 'r-', label='Trendline', color=blue)
plt.xlabel("Shoulder HA at FP Relative to Mean")
plt.ylabel("Predicted Throwing Velocity (MPH)")
plt.title("Shoulder Horizontal Abduction at Foot Plant Partial Dependence")
plt.tight_layout()
plt.grid(True)
plt.show()

# Shoulder HA at FP specific values for reference
print(poi_metrics['shoulder_horizontal_abduction_fp'].mean())
print(poi_metrics['shoulder_horizontal_abduction_fp'].std())

# Define the groups/categories and their corresponding feature indices
groups = {
    'Arm Action': [4, 5, 6, 7, 9, 10, 11, 12, 24, 25, 26, 27, 28, 32, 33, 41, 43, 44, 45, 46, 47, 48, 49],
    'Arm Velos': [0, 1],
    'Torso/Rotation': [2, 15, 16, 17, 22, 29, 30, 31, 34, 35, 36, 42, 63],
    'Pelvis/Hip': [3, 8, 18, 19, 20, 23, 50, 51, 56, 57, 58, 62],
    'Lead Leg': [12, 13, 14, 37, 39, 40, 52, 53, 54, 55, 69, 70, 71, 72, 73],
    'Rear Leg': [59, 60, 61, 64, 65, 66, 67, 68, 74, 75],
    'COG': [21, 38]
}

# Calculate the total feature importance and average importance for each group
group_importance = {}
for group_name, group_indices in groups.items():
    group_importance[group_name] = np.mean(feature_importance[group_indices])

# Sort the groups based on average importance in descending order
sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
sorted_group_names = [group[0] for group in sorted_groups]
sorted_group_importance = [group[1] for group in sorted_groups]

blue = (169/255, 196/255, 235/255, 1)
plt.rcParams.update({'font.size': 20})
# Plot feature importance by group
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_group_importance)), sorted_group_importance, color=blue)
plt.xticks(range(len(sorted_group_importance)), sorted_group_names, rotation=30)
plt.xlabel('Category')
plt.ylabel('Average Importance')
plt.title('Feature Importance by Category')
plt.yticks([])
plt.show()

# Define the feature index for shoulder horizontal abduction at foot plant
feature_index = 64

# Extract the feature from the test features
feature_values = test_features[:, feature_index].detach().numpy()

# Create an empty array to store the predicted outputs
predicted_outputs = np.zeros_like(feature_values)

# Compute the predictions for different feature values
for i, value in enumerate(feature_values):
    test_features_copy = test_features.clone()
    test_features_copy[:, feature_index] = torch.tensor(value)
    outputs = model(test_features_copy)
    predicted_outputs[i] = outputs.mean().item()  # Store the mean prediction
    
# Fit a polynomial regression model to the data
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(feature_values.reshape(-1, 1))
regressor = LinearRegression()
regressor.fit(X_poly, predicted_outputs)

# Predict outputs for the range of feature values
feature_values_range = np.linspace(np.min(feature_values), np.max(feature_values), 100)
X_poly_range = poly_features.transform(feature_values_range.reshape(-1, 1))
predicted_outputs_range = regressor.predict(X_poly_range)

blue = (169/255, 196/255, 235/255, 1)

# Plot the partial dependence and trendline
plt.figure(figsize=(10, 6))
plt.scatter(feature_values, predicted_outputs)
plt.plot(feature_values_range, predicted_outputs_range, 'r-', label='Trendline', color=blue)
plt.xlabel("Max Rear Leg GRF Relative to Mean")
plt.ylabel("Predicted Throwing Velocity (MPH)")
plt.title("Max Rear Leg GRF Partial Dependence")
plt.yticks(np.arange(84, 90, 2))
plt.tight_layout()
plt.grid(True)
plt.show()
