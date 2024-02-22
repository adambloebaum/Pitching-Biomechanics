import torch
import torch.nn as nn
from lime import lime_tabular
import pandas as pd
import joblib

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
        
        # Dropout function
        self.dropout = nn.Dropout(0.11027576557220337)

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

# Load the test data
df = pd.read_csv('X_test_scaled.csv')
feature_names = df.columns.tolist()
X_test_scaled = df.values

y_test = pd.read_csv('y_test.csv')

# Load the trained model
model = PitchVelocityModel(X_test_scaled.shape[1])
model.load_state_dict(torch.load('pitch_velocity_model.pth'))
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pk1')

# Function to make predictions with the PyTorch model
def pytorch_predict(input_data):
    # Convert input data to PyTorch tensor
    model_input = torch.tensor(input_data, dtype=torch.float32)
    # Model prediction
    model.eval()
    with torch.no_grad():
        predictions = model(model_input).numpy()
    return predictions

# Create a LIME explainer object
explainer = lime_tabular.LimeTabularExplainer(
    training_data=scaler.inverse_transform(X_test_scaled),
    feature_names=feature_names,
    mode='regression'
)

# Select the instance you want to explain
instance_index = 0
instance = X_test_scaled[instance_index]

# Explain the model's prediction on this instance
exp = explainer.explain_instance(
    data_row=instance, 
    predict_fn=pytorch_predict
)

# Get the model's prediction and actual value
predicted_value = pytorch_predict(instance.reshape(1, -1))
actual_value = y_test.iloc[instance_index].item()

# Show the explanation
exp.show_in_notebook(show_table=True)

fig = exp.as_pyplot_figure()
fig.suptitle(f'Predicted Value: {predicted_value[0][0]:.2f}, Actual Value: {actual_value}')
fig.savefig('lime_explanation.png', bbox_inches='tight')
