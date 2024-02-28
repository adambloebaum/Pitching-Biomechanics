# biomech

## deep learning
This project involves the development of a neural network model for predicting pitch velocity in baseball. The project includes three main components:
1. `neural_net.py`: This script includes data handling (including database interactions and data preprocessing), model creation using PyTorch, and training processes.
3. `optuna_tuning.py`: This script focuses on model tuning using Optuna, a hyperparameter optimization framework, along with data handling and model evaluation.
2. `lime_explanation.py`: This script implement a model class, `PitchVelocityModel` from `neural_net.py`, using PyTorch, and uses LIME (Local Interpretable Model-agnostic Explanations) for explaining predictions.

### Features
- Neural network-based prediction of pitch velocity.
- Database integration for data handling.
- Model explanation using LIME.
- Hyperparameter tuning using Optuna.

### Configuration
To run this project, you need to have Python installed on your system along with several dependencies. The project is developed with CUDA 12.3.1 and CuDNN 9.0 for GPU acceleration. Here are the steps to set up your environment:

1. **Python Installation**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **CUDA and CuDNN Setup**: Install CUDA 12.3.1 and CuDNN 9.0. These are necessary for GPU acceleration. You can download them from NVIDIA's official website. Ensure that your GPU is compatible with these versions.

3. **Install Required Libraries**: The project requires several Python libraries. You can install them using pip. The required libraries include:
   - PyTorch: For neural network implementation. Install it with the command suitable for your CUDA version from [PyTorch's official website](https://pytorch.org/).
   - Pandas, Numpy: For data manipulation.
   - MySQL Connector: For database operations.
   - Joblib: For saving models.
   - Optuna: For hyperparameter optimization.
   - Scikit-learn: For data preprocessing and metrics.
   - Matplotlib: For plotting graphs.
   - LIME: For generating local interpretable model-agnostic explanations.

   You can install these libraries using the following command:
   ```
   pip install torch pandas numpy mysql-connector-python joblib optuna scikit-learn matplotlib lime
   ```

4. **Clone the Repository**: Clone the project repository to your local machine or download the project files.

5. **Database Setup**: The script uses a private database connection to retrieve the data, but it can be find at the repo referenced in `.bib`

### Usage
Run `neural_net.py` to train and evaluate the model.
Run `optuna_tuning.py` to fine-tune model hyperparameters.
Run `lime_explainer.py` to create model explanation visualizations.

### Files Description

#### `neural_net.py`
This script is the core of the project, where the neural network model is defined, trained, and evaluated.

```python
class PitchVelocityModel(nn.Module):
    def __init__(self, num_features):
        super(PitchVelocityModel, self).__init__()
        # Fully connected layers
        self.fc1 = torch.nn.Linear(num_features, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        # ... additional layers ...
        self.fc7 = torch.nn.Linear(32, 1)
        # Dropout function
        self.dropout = nn.Dropout(0.11027576557220337)
        # Relu activation function
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # Forward pass logic
        return x

# ... additional code for data handling, training, and evaluation ...
```

This file handles data preprocessing, model training, validation, and testing. It includes database interaction for data retrieval, feature scaling, model evaluation using mean squared error, and saving the trained model.

#### `lime_explanation.py`
This script uses the LIME package to create interpretable explanations for the neural network's predictions.

```python
# Load the trained model
model = PitchVelocityModel(X_test_scaled.shape[1])
model.load_state_dict(torch.load('pitch_velocity_model.pth'))
model.eval()

# Create a LIME explainer object
explainer = lime_tabular.LimeTabularExplainer(
    training_data=scaler.inverse_transform(X_test_scaled),
    feature_names=feature_names,
    mode='regression'
)

# Explain the model's prediction on an instance
exp = explainer.explain_instance(
    data_row=instance, 
    predict_fn=pytorch_predict
)
```

It involves loading the trained model, setting up a LIME explainer, and generating explanations for specific instances.

#### `optuna_tuning.py`
This script focuses on optimizing the model parameters using the Optuna framework.

```python
def objective(trial):
    # Hyperparameters to be tuned
    batch_size = trial.suggest_categorical('batch_size', [64, 32])
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
    # ... additional parameter suggestions ...

    # Model setup
    model = PitchVelocityModel(X_train_scaled.shape[1], dropout_prob)
    # ... training loop ...

    return mean_val_loss

# Creating the Optuna study object
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=400)
```

This file includes the definition of the Optuna objective function and the execution of the optimization study, aiming to find the best hyperparameters for the model.

### Model Design and Training
- **Feature Engineering**: New features are engineered relative to body weight, enhancing the predictive power of the model.
- **Data Scaling**: All features are standard scaled to ensure uniformity and better convergence during training.
- **Data Splitting**: The dataset is split into an 80-10-10 distribution for training, validation, and testing, ensuring a comprehensive evaluation.
- **Model Architecture**: The regression model consists of linear layers of various sizes, integrated with ReLU activation functions and two dropout layers to introduce regularization.
- **GPU Acceleration**: Training is accelerated using GPU capabilities, significantly reducing the model's training time.
- **Loss Function and Optimizer**: A mean squared error loss function is used in conjunction with the Adam optimizer, striking a balance between efficient convergence and model accuracy.
- **Preventing Overfitting**: Early stopping, dropout, and a learning rate scheduler are implemented as strategies to combat overfitting.
- **Training and Validation Monitoring**: During the training and validation phases, both training loss and validation loss are displayed. Additionally, training loss over epochs is plotted for visual assessment, and mean squared error (MSE) is reported after the testing loop.

### Hyperparameter Optimization
- **Trial and Error**: Initial model parameters, including the architecture, loss function, and optimizer, are determined through trial and error.
- **Optuna Framework**: For fine-tuning, the Optuna framework is employed. It optimizes learning rate, batch size, and dropout probability through an extensive study involving various trials.
- **Pruning for Efficiency**: Pruning techniques are utilized within the Optuna framework to optimize computation time and GPU resource usage.

### Model Interpretation with LIME
- **LIME Integration**: The Local Interpretable Model-agnostic Explanations (LIME) package is employed to create a local, interpretable model that approximates the neural network's behavior.
- **Explanation of Predictions**: Using LIME, the model's prediction process is demystified, particularly by focusing on the first value in the test set to generate a local explainer model.

### Model Performance Metrics
- **Test MSE**: The model achieves a test mean squared error of 10.79, indicating that, on average, the model's predictions are off by approximately 3.28 mph.
- **Analysis of Residual Error**: The residual error can be attributed to factors beyond the model's scope, such as physical performance attributes like strength and power output, as well as the inherent variability in throwing.

## Dashboards