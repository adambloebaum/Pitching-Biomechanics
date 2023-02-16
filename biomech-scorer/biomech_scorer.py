import dash
import dash_html_components as html
import dash_core_components as dcc
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
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# arm action index
aa_index = [5, 6, 7, 8, 10, 11, 12, 25, 26, 27, 28, 32, 33, 41, 43, 44, 45, 46, 47, 48, 49]
# arm velos index
av_index = [1, 2]
# torso/rotation index
rot_index = [3, 16, 17, 18, 23, 29, 30, 31, 34, 35, 36, 42]
# hip-shoulder sep/hip index
hss_index = [4, 9, 19, 20, 21, 24, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63]
# lead leg block index
block_index = [13, 14, 15, 37, 69, 70, 71, 72, 73, 75]
# cog index
cog_index = [22, 38, 39, 40, 59, 60, 61, 64, 65, 66, 67, 68, 74]

# establishing global variables
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('Biomech Scorer', external_stylesheets=external_stylesheets)
labels = ['Arm Action', 'Arm Velos', 'Torso', 'Pelvis', 'Lead Leg Block', 'COG']
default_fig = px.line_polar(r=[0, 0, 0, 0, 0, 0], theta=labels, line_close=True, range_r=[0, 100])
allowed_extensions = {'csv', 'xls'}
required_shape = (1, 81)

def create_scores(df):
    df = df.dropna()
    df = df.select_dtypes(exclude=['object'])
    
    percentiles = []
    
    for metric in metrics:
        population = poi_metrics[metric]
        individual = df[metric].item()
        percentile = round(scipy.stats.percentileofscore(population, individual))
        percentiles.append(percentile)
    
    del percentiles[0]
    del percentiles[0]
    
    r_squared_values = []
    
    for metric in metrics:
        y = poi_metrics['pitch_speed_mph']
        X = poi_metrics[metric]
        corr_matrix = np.corrcoef(y, X)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        r_squared_values.append(R_sq)
    
    del r_squared_values[0]
    del r_squared_values[0]

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

    rot_percentiles = []
    rot_rsq = []
    rot_composite = []
    
    for index in rot_index:
        rot_percentiles.append(percentiles[index])
        rot_rsq.append(r_squared_values[index])
    
    rot_weights = rot_rsq / sum(rot_rsq)
    
    for i in range(len(rot_weights)):
        num = rot_weights[i] * rot_percentiles[i]
        rot_composite.append(num)
    
    rot_score = round(sum(rot_composite))

    hss_percentiles = []
    hss_rsq = []
    hss_composite = []
    
    for index in hss_index:
        hss_percentiles.append(percentiles[index])
        hss_rsq.append(r_squared_values[index])
    
    hss_weights = hss_rsq / sum(hss_rsq)
    
    for i in range(len(hss_weights)):
        num = hss_weights[i] * hss_percentiles[i]
        hss_composite.append(num)
    
    hss_score = round(sum(hss_composite))

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

    scores = [aa_score, av_score, rot_score, hss_score, block_score, cog_score]
    
    return scores

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def create_radial_plot():
    return default_fig
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
                    df = pd.read_csv(io.BytesIO(decoded))
            except Exception as e:
                return {}
            if df.shape == required_shape:
                scores = create_scores(df)
                fig = px.line_polar(r=scores, theta=labels, line_close=True, range_r=[0, 100])
                fig.update_traces(fill='toself')
                return fig
            else:
                return default_fig
        else:
            return default_fig
    else:
        return default_fig

app.layout = html.Div([
    html.H1('Biomechanics Scorer', style={'textAlign': 'center'}),
    dcc.Upload(html.Button('Upload File'), id='upload-data', multiple=False, style={'textAlign': 'center'}),
    dcc.Graph(id='radial-plot', figure=create_radial_plot(), style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
