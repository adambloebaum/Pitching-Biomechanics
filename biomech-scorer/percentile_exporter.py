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

# looping through metrics and assigning percentile values in the dict
for metric in metrics:
    population = poi_metrics[metric]
    individual = sample_metrics[metric].item()
    percentile = round(scipy.stats.percentileofscore(population, individual))
    results[metric] = percentile

# sort by percentile and drop session column
del results['session']
sorted_results = sorted(results.items(), key=lambda x :x[1], reverse=True)

# export result as csv
with open('sample_percentiles.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sorted_results)
