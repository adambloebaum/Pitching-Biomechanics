import pandas as pd
import mysql.connector

'''# configure database connection
db_config = {
    "host": "HOST",
    "user": "USER",
    "password": "PASSWORD",
    "database": "DATABASE"
}

# connect to db
connection = mysql.connector.connect(**db_config)

# set up query
query = 'SELECT * FROM table;'

# execute query
df = pd.read_sql(query, connection)

# close connection
connection.close()'''

db_config = {
    "host": "10.200.200.107",
    "user": "scriptuser1",
    "password": "YabinMarshed2023@#$",
    "database": "biomech_pitching_db"
}
connection = mysql.connector.connect(**db_config)
query = 'SELECT * FROM bp_poi_metrics INNER JOIN bp_sessions ON bp_poi_metrics.session = bp_sessions.session;'
df = pd.read_sql(query, connection)
connection.close()

# drop rows with na values
df.dropna()

# save df as csv
df.to_csv('df.csv', index=False)

# select a random row
random_row = df.sample()

# save row as csv
random_row.to_csv('sample_poi_metrics.csv', index=False)
