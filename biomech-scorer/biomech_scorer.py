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

# empty dict
results = {}

# loop through metrics and find percentile values in the dict
for metric in metrics:
    population = poi_metrics[metric]
    individual = sample_metrics[metric].item()
    percentile = round(scipy.stats.percentileofscore(population, individual))
    results[metric] = percentile

# sort by percentile and drop session column
sorted_results = sorted(results.items(), key=lambda x :x[1], reverse=True)
del results['session']

# export result as csv
with open('sample_percentiles.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sorted_results)

# empty list
r_squared_values = []

# loop through metrics and find each Rsquared value with pitch speed
for metric in metrics:
    y = poi_metrics['pitch_speed_mph']
    X = poi_metrics[metric]
    corr_matrix = np.corrcoef(y, X)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    r_squared_values.append(R_sq)

# empty list
percentiles = []

# loop through metrics and find each metric's percentile ranking
for metric in metrics:
    population = poi_metrics[metric]
    individual = sample_metrics[metric].item()
    percentile = round(scipy.stats.percentileofscore(population, individual))
    percentiles.append(percentile)

# remove session and pitch speed values 
del r_squared_values[0]
del r_squared_values[0]
del percentiles[0]
del percentiles[0]

# scale percentiles to 0 to 1.0
basis_points = [x / 100 for x in percentiles]

# scale Rsquared values relative to sum
weights = r_squared_values / sum(r_squared_values)

# empty list
composite = []

# loop through weights and basis_points and find product
for weight in weights:
    for point in basis_points:
        composite.append(weight * point)

# sum product
score = round(sum(composite))

print("Your composite biomechanics score is: " + str(score))
