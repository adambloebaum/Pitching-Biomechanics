import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.keras import layers
import numpy as np
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

app = dash.Dash('Pitching Velocity Predictor')

@app.callback(
    Output('output', 'children'),
    Input('shafp_input', 'value'),
    Input('mser_input', 'value'),
    Input('msirv_input', 'value'),
    Input('mtrv_input', 'value'),
    Input('rhssafp_input', 'value'))
def update_velo(shafp, mser, msirv, mtrv, rhssafp):
    values = [shafp, mser, msirv, mtrv, rhssafp]
    values = [float(x) for x in values]
    values = np.array(values).reshape((1, len(values)))
    values = values.astype('float32')
    prediction = dnn_multi.predict(values)
    return str(prediction)

# app layout
app.layout = html.Div([
        # title
        dcc.Markdown('PITCHING VELOCITY PREDICTOR', style={'textAlign': 'center', 'border': '2px black solid'}),
        
        # input labels
        dcc.Markdown('Shoulder Horizontal Abduction at FP:', style={'position': 'absolute', 'top': 60, 'left': 80, 'font-weight': 'bold'}),
        dcc.Markdown('Max Shoulder External Rotation:', style={'position': 'absolute', 'top': 140, 'left': 80, 'font-weight': 'bold'}),
        dcc.Markdown('Max Shoulder Internal Rotational Velo:', style={'position': 'absolute', 'top': 220, 'left': 80, 'font-weight': 'bold'}),
        dcc.Markdown('Max Torso Rotational Velo:', style={'position': 'absolute', 'top': 300, 'left': 80, 'font-weight': 'bold'}),
        dcc.Markdown('Rotational Hip Shoulder Separation at FP:', style={'position': 'absolute', 'top': 380, 'left': 80, 'font-weight': 'bold'}),
        dcc.Markdown('Predicted Pitching Velocity:', style={'position': 'absolute', 'top': 140, 'right': 400, 'font-weight': 'bold'}),
        
        # input boxes
        dcc.Input(id='shafp_input', type='number', placeholder='Ex: 35', value=35, required=True, style={'position': 'absolute', 'top': 100, 'left': 80}),
        dcc.Input(id='mser_input', type='number', placeholder='Ex: 140', value=140, required=True, style={'position': 'absolute', 'top': 180, 'left': 80}),
        dcc.Input(id='msirv_input', type='number', placeholder='Ex:4000', value=4000, required=True, style={'position': 'absolute', 'top': 260, 'left': 80}),
        dcc.Input(id='mtrv_input', type='number', placeholder='Ex: 1000', value=1000, required=True, style={'position': 'absolute', 'top': 340, 'left': 80}),
        dcc.Input(id='rhssafp_input', type='number', placeholder='Ex: 30', value=30, required=True, style={'position': 'absolute', 'top': 420, 'left': 80}),
        
        # output box
        html.Div(id='output', style={'position': 'absolute', 'top': 155, 'right': 320, 'font-weight': 'bold'})
    ])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
