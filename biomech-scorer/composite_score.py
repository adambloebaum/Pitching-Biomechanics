import pandas as pd
import statsmodels
import scipy
from scipy import stats
import numpy as np
import csv

# load in data
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)
sample_metrics = pd.read_csv('sample_metrics.csv', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

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

weights = r_squared_values / sum(r_squared_values)

percentiles = []

for metric in metrics:
    population = poi_metrics[metric]
    individual = sample_metrics[metric].item()
    percentile = round(scipy.stats.percentileofscore(population, individual))
    percentiles.append(percentile)

del percentiles[0]
del percentiles[0]

composite=[]
for i in range(len(weights)):
    num = weights[i] * percentiles[i]
    composite.append(num)

score = round(sum(composite))

print("Your composite biomechanics score is: " + str(score))
