import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

app = dash.Dash('Biomech Vis')

# load in data
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# scatter plot
def create_scatter_plot():
    fig = px.scatter(poi_metrics, x='pitch_speed_mph', y=metrics[2], trendline='ols')
    return fig
@app.callback(
    Output('scatter_plot', 'figure'),
    Input('metric_dropdown','value'))
def update_scatter_plot(value):
    fig = px.scatter(poi_metrics, x='pitch_speed_mph', y=value, trendline='ols')
    return fig

# histogram
def create_histogram():
    fig = px.histogram(poi_metrics, x=metrics[2])
    return fig
@app.callback(
    Output('histogram', 'figure'),
    Input('metric_dropdown_2', 'value'))
def update_histogram(value):
    fig = px.histogram(poi_metrics, x=poi_metrics[value])
    return fig
    
# app layout
app.layout = html.Div(children=[
        # title
        html.H1('BIOMECH VIS', style={'textAlign': 'center', 'border': '2px black solid'}),
        
        # dropdown labels
        html.Div([
            dcc.Markdown('Scatter Plot Y-Axis:', style={'position': 'absolute', 'top': 55, 'left': 100, 'font-weight': 'bold'}),
            dcc.Markdown('Histogram X-Axis:', style={'position': 'absolute', 'top': 55, 'right': 420, 'font-weight': 'bold'})
            ]),
        
        # dropdowns
        html.Div([
            dcc.Dropdown(
                id = 'metric_dropdown',
                options = [{'label':metric, 'value':metric} for metric in metrics],
                value = metrics[2],
                multi=False,
                clearable=False,
                style={'width': '50%', 'position': 'absolute', 'top': 50, 'left': 50}),
            dcc.Dropdown(
                id = 'metric_dropdown_2',
                options = [{'label':metric, 'value':metric} for metric in metrics],
                value = metrics[2],
                multi=False,
                clearable=False,
                style={'width': '50%', 'position': 'absolute', 'top': 50, 'right': 116})
            ]),
        
        # graphs
        html.Div([
            dcc.Graph(
                id='scatter_plot', figure=create_scatter_plot(), style={'width': '48%', 'position': 'absolute', 'top': 140, 'left': 20}),
            dcc.Graph(
                id='histogram', figure=create_histogram(), style={'width': '48%', 'position': 'absolute', 'top': 140, 'right': 20})
            ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
