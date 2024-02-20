import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output, State
import io
import pandas as pd
import statsmodels
import scipy
from scipy import stats
import numpy as np
import plotly.express as px
import base64

# load in data
poi_metrics = pd.read_csv(r'PATH TO FILE', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# arm action index
aa_index = [6, 7, 8, 9, 11, 12, 13, 27, 28, 29, 30, 31, 35, 36, 44, 46, 47, 48, 49, 50, 51, 52]
# arm velos index
av_index = [2, 3]
# torso index
torso_index = [4, 18, 19, 20, 25, 32, 33, 34, 37, 38, 39, 45, 66]
# hip index
hip_index = [5, 10, 21, 22, 23, 26, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
# lead leg block index
block_index = [14, 15, 16, 17, 40, 41, 42, 43]
# cog index
cog_index = [24]

# establish variables
app = dash.Dash('Biomech Comp Scorer')
labels = ['Arm Action', 'Arm Velos', 'Torso', 'Pelvis', 'Lead Leg Block', 'COG']
allowed_extensions = {'csv', 'xls'}
required_shape = (1, 79)

def create_scores(df):
    df = df.dropna()
    df = df.select_dtypes(exclude=['object'])
    
    percentiles = []
    
    for metric in metrics:
        population = poi_metrics[metric]
        individual = df[metric].item()
        percentile = round(scipy.stats.percentileofscore(population, individual))
        percentiles.append(percentile)
    
    r_squared_values = []
    
    for metric in metrics:
        y = poi_metrics['pitch_speed_mph']
        X = poi_metrics[metric]
        corr_matrix = np.corrcoef(y, X)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        r_squared_values.append(R_sq)

    # arm action scoring
    aa_percentiles = []
    aa_rsq = []
    aa_composite = []
    
    for index in aa_index:
        aa_percentiles.append(percentiles[index])
        aa_rsq.append(r_squared_values[index])
    
    aa_weights = aa_rsq / sum(aa_rsq)
    
    for i in range(len(aa_weights)):
        num = aa_weights[i] * aa_percentiles[i]
        aa_composite.append(num)
    
    aa_score = round(sum(aa_composite))

    # arm velos scoring
    av_percentiles = []
    av_rsq = []
    av_composite = []
    
    for index in av_index:
        av_percentiles.append(percentiles[index])
        av_rsq.append(r_squared_values[index])
    
    av_weights = av_rsq / sum(av_rsq)
    
    for i in range(len(av_weights)):
        num = av_weights[i] * av_percentiles[i]
        av_composite.append(num)
    
    av_score = round(sum(av_composite))

    # torso scoring
    torso_percentiles = []
    torso_rsq = []
    torso_composite = []
    
    for index in torso_index:
        torso_percentiles.append(percentiles[index])
        torso_rsq.append(r_squared_values[index])
    
    torso_weights = torso_rsq / sum(torso_rsq)
    
    for i in range(len(torso_weights)):
        num = torso_weights[i] * torso_percentiles[i]
        torso_composite.append(num)
    
    torso_score = round(sum(torso_composite))

    # hip/shoulder separation scoring
    hip_percentiles = []
    hip_rsq = []
    hip_composite = []
    
    for index in hip_index:
        hip_percentiles.append(percentiles[index])
        hip_rsq.append(r_squared_values[index])
    
    hip_weights = hip_rsq / sum(hip_rsq)
    
    for i in range(len(hip_weights)):
        num = hip_weights[i] * hip_percentiles[i]
        hip_composite.append(num)
    
    hip_score = round(sum(hip_composite))

    # lead leg block scoring
    block_percentiles = []
    block_rsq = []
    block_composite = []
    
    for index in block_index:
        block_percentiles.append(percentiles[index])
        block_rsq.append(r_squared_values[index])
    
    block_weights = block_rsq / sum(block_rsq)
    
    for i in range(len(block_weights)):
        num = block_weights[i] * block_percentiles[i]
        block_composite.append(num)
    
    block_score = round(sum(block_composite))

    # center of gravity velo scoring
    cog_percentiles = []
    cog_rsq = []
    cog_composite = []
    
    for index in cog_index:
        cog_percentiles.append(percentiles[index])
        cog_rsq.append(r_squared_values[index])
    
    cog_weights = cog_rsq / sum(cog_rsq)
    
    for i in range(len(cog_weights)):
        num = cog_weights[i] * cog_percentiles[i]
        cog_composite.append(num)
    
    cog_score = round(sum(cog_composite))

    scores = [aa_score, av_score, torso_score, hip_score, block_score, cog_score]
    
    return scores

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def create_radial_plot():
    fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig
@app.callback(
    Output('radial-plot', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def update_radial_plot(contents, filename):
    if contents is not None:
        if allowed_file(filename):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    df = pd.read_excel(io.BytesIO(decoded))
            except Exception as e:
                fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                )
                return fig

            # preprocessing
            df = df.dropna()
            df = df.select_dtypes(exclude=['object'])

            if df.shape[1] == 79:
                scores = create_scores(df)
                fig = px.line_polar(r=scores, theta=labels, line_close=True, range_r=[0, 100])
                fig.update_traces(fill='toself')
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )
                return fig
            else:
                fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                )
                return fig
        else:
            fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
            fig.update_layout(
                paper_bgcolor='white',
                plot_bgcolor='white',
            )
            return fig
    else:
        fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
        )
        return fig

app.layout = html.Div([
    # Custom Header
    html.H1(
        'Pitching Biomechanics Composite Score',
        style={
            'textAlign': 'center',
            'color': 'white',
            'backgroundColor': '#333333',
            'padding': '10px',
            'fontFamily': 'Calibri',
            'fontSize': '32 px',
            'fontWeight': 'bold',
            'margin': '10',
            'width': '100%'
        }
    ),
    
    # Upload Button
    dcc.Upload(
        id='upload-data',
        children=html.Button(
            'Upload File',
            id='upload-button',
            style={
                'backgroundColor': 'blue',
                'color': 'white',
                'borderRadius': '5px',
                'padding': '10px 20px',
                'border': 'none',
                'cursor': 'pointer',
                'fontSize': '24px',
                'fontWeight': 'bold',
                'outline': 'none',
                'boxShadow': 'none',
                'height': '50px',
                'lineHeight': '25px',
                'fontFamily': 'Calibri'
            }
        ),
        multiple=False,
        style={
            'textAlign': 'center',
            'display': 'inline-block',
            'margin': 'auto',
        }
    ),

    # Polar Chart
    dcc.Graph(
        id='radial-plot',
        figure=create_radial_plot(),
        config={'displayModeBar': False},
        style={
            'height': '700px',
            'width': '100%',
            'display': 'flex',
            'justifyContent': 'center',
        }
    )
], style={'textAlign': 'center', 'fontFamily': 'Calibri'})

if __name__ == '__main__':
    app.run_server(debug=True)
