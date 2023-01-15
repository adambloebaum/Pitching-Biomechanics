import numpy as np
import pandas as pd
np.set_printoptions(precision=3, suppress=True)

# load in data
poi_metrics = pd.read_csv('poi_metrics.csv', index_col=False)

# preprocessing
poi_metrics = poi_metrics.dropna()
poi_metrics = poi_metrics.drop('session', axis=1)
poi_metrics = poi_metrics.select_dtypes(exclude=['object'])
poi_metrics_columns = poi_metrics.columns
metrics = list(poi_metrics.columns)

# empty dictionary
r_squared_values = {}

# calculate r squared values
for metric in metrics:
    y = poi_metrics['pitch_speed_mph']
    X = poi_metrics[metric]
    corr_matrix = np.corrcoef(y, X)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    pair = {str(metric): R_sq}
    r_squared_values.update(pair)
    
# sort and print
sorted_values = sorted(r_squared_values.items(), key=lambda x:x[1], reverse=True)
print(*sorted_values, sep='\n')
