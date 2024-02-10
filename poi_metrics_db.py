import pandas as pd
import mysql.connector

# configure database connection
db_config = {
    "host": "HOST",
    "user": "USER",
    "password": "PASSWORD",
    "database": "DATABASE"
}

# connect to db
connection = mysql.connector.connect(**db_config)

# set up query
query = 'SELECT * FROM TABLE;'

# execute query
df = pd.read_sql(query, connection)

# close connection
connection.close()

# drop rows with na values
df.dropna()

# save df as csv
df.to_csv('biomech_pitching_poi_metrics.csv', index=False)

# select a random row
random_row = df.sample()

# save row as csv
random_row.to_csv('sample_poi_metrics.csv', index=False)