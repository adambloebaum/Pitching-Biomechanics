import dash
from dash import html, dcc
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import statsmodels.api as sm

app = dash.Dash('Biomech Viewer')

# load in data
poi_metrics = pd.read_csv('PATH TO FILE', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics = poi_metrics[poi_metrics['pitch_speed_mph'] < 110]
poi_metrics = poi_metrics[poi_metrics['pitch_speed_mph'] > 40]
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# scatter plot
def create_scatter_plot():
    fig = px.scatter(poi_metrics, x='pitch_speed_mph', y=metrics[2], trendline='ols')
    return fig
@app.callback(
    [Output('scatter_plot', 'figure'),
    Output('r_squared_value', 'children')],
    Input('metric_dropdown','value'))
def update_scatter_plot(value):
    fig = px.scatter(poi_metrics, x='pitch_speed_mph', y=value, trendline='ols')
    # Calculate R-squared value
    X = poi_metrics['pitch_speed_mph']
    y = poi_metrics[value]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    r_squared = model.rsquared
    return fig, f"R Squared Value: {r_squared:.3f}"

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
        # Custom Header
        html.H1(
            'Pitching Biomech Viewer',
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
        
        # dropdown labels
        html.Div([
            dcc.Markdown('Scatter Plot Y-Axis:', style={'position': 'absolute', 'top': 100, 'left': 100, 'font-weight': 'bold', 'fontFamily': 'Calibri'}),
            dcc.Markdown('Histogram X-Axis:', style={'position': 'absolute', 'top': 100, 'right': 740, 'font-weight': 'bold', 'fontFamily': 'Calibri'})
            ]),
        
        # rsquared text box
        html.Div(id='r_squared_value', style={'position': 'absolute', 'top': 118, 'left': 560, 'font-weight': 'bold', 'fontFamily': 'Calibri'}),

        # dropdowns
        html.Div([
            dcc.Dropdown(
                id = 'metric_dropdown',
                options = [{'label':metric, 'value':metric} for metric in metrics],
                value = metrics[2],
                multi=False,
                clearable=False,
                style={'width': '40%', 'position': 'absolute', 'top': 55, 'left': 120, 'fontFamily': 'Calibri'}),
            dcc.Dropdown(
                id = 'metric_dropdown_2',
                options = [{'label':metric, 'value':metric} for metric in metrics],
                value = metrics[2],
                multi=False,
                clearable=False,
                style={'width': '40%', 'position': 'absolute', 'top': 55, 'right': 210, 'fontFamily': 'Calibri'})
            ]),
        
        # graphs
        html.Div([
            dcc.Graph(
                id='scatter_plot', figure=create_scatter_plot(), style={'width': '48%', 'position': 'absolute', 'top': 150, 'left': 20}, config={'displayModeBar': False}),
            dcc.Graph(
                id='histogram', figure=create_histogram(), style={'width': '48%', 'position': 'absolute', 'top': 150, 'right': 20}, config={'displayModeBar': False})
            ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
