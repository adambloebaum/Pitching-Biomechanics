import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import statsmodels
np.set_printoptions(precision=3, suppress=True)

# load in data
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.drop('session', axis=1)
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# select 5 metrics
big5 = poi_metrics[['pitch_speed_mph', 'shoulder_horizontal_abduction_fp', 'max_shoulder_external_rotation', 'max_shoulder_internal_rotational_velo', 'max_torso_rotational_velo', 'rotation_hip_shoulder_separation_fp']]

#training and test sets
train_big5 = big5.sample(frac=0.8, random_state=0)
test_big5 = big5.drop(train_big5.index)

#splitting features from labels
train_features = train_big5.copy()
test_features = test_big5.copy()
train_labels = train_features.pop('pitch_speed_mph')
test_labels = test_features.pop('pitch_speed_mph')

#build normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# deep neural network with multiple inputs
dnn_multi = tf.keras.Sequential([
    normalizer,
    ks.layers.Dense(64, activation='relu'),
    ks.layers.Dense(64, activation='relu'),
    ks.layers.Dense(64, activation='relu'),
    ks.layers.Dense(1)
])

dnn_multi.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_absolute_error')

history = dnn_multi.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=200)

# plot loss function
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,20])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPH]')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    
plot_loss(history)
