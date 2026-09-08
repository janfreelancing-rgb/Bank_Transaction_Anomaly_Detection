import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from unicodedata import category

# Upload dataset

df = pd.read_csv('bank_transactions_data_2_augmented_clean_2.csv')



# Basic Data Inspection

#print('\n Shape of Dataset:', df.shape)
#print('\n Data Types of Dataset:', df.dtypes)
#print('\n Number of rows:', df.shape[0])
#print('\n Total overview of Dataset:', df.describe())
#print('\n All Information about Dataset:', df.info)
#print('\n Looking for duplicate values:', df.duplicated().sum())
#print('\n Looking for missing values:', df.isnull().sum())
#print('\n Number of columns:', df.columns.value_counts())
#print('\n Names of Columns:\n', df.columns)
    #['TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate','TransactionType', 'Location', 'DeviceID',
    # 'IP Address', 'MerchantID','Channel', 'CustomerAge', 'CustomerOccupation', 'TransactionDuration','LoginAttempts', 'AccountBalance']


# Data Pre-processing

# Handling Transaction Date in correct way
print('\n data types before date time converting:',df.dtypes)
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors = 'coerce')
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionYear'] = df['TransactionDate'].dt.year
print('\n data types after date time converting:',df.dtypes)
#print('\n Transaction Hour:',df['TransactionHour'])
#print('\n Transaction Day:',df['TransactionDay'])
#print('\n Transaction Month:',df['TransactionMonth'])
#print('\n Year of Transaction:',df['TransactionYear'])


# Drop original column 'TransactionDate'
df = df.drop(['TransactionDate'], axis = 1)
#print('\n New columns formation:',df.columns.values)
#   [      'TransactionID',           'AccountID',   'TransactionAmount',
#      'TransactionType',            'Location',            'DeviceID',
#           'IP Address',          'MerchantID',             'Channel',
#          'CustomerAge',  'CustomerOccupation', 'TransactionDuration',
#        'LoginAttempts',      'AccountBalance',     'TransactionHour',
#       'TransactionDay',    'TransactionMonth',     'TransactionYear']

#print('\n info:',df.info)
#print('\n description:',df.describe())


# Remove High cardinality IDs features
columns_drop = ['TransactionID','AccountID','DeviceID', 'IP Address', 'MerchantID' ]

df = df.drop(columns = columns_drop, errors = 'ignore')
#print('\n New columns formation:',df.columns.values)
#   [  'TransactionAmount',     'TransactionType',            'Location',
#              'Channel',         'CustomerAge',  'CustomerOccupation',
#  'TransactionDuration',       'LoginAttempts',      'AccountBalance',
#   [ ]  'TransactionHour',      'TransactionDay',    'TransactionMonth', 'TransactionYear']


# Remove duplicates
print('\n shape before remove duplicates:',df.shape)
df = df.drop_duplicates()
print('\n shape after removed duplicates:',df.shape)

# Handle missing values
print('\n Null Values:\n',df.isnull().sum())
#print('\n Numeric values columns:\n')
numeric_values = df.select_dtypes(include = ['int64', 'float64']).columns
#print(numeric_values)
df[numeric_values] = df[numeric_values].fillna(df[numeric_values].median())
print('\n Fill N.A values with median:\n',df.isnull().sum())


# Categorical Columns
categorical_columns = df.select_dtypes(include = ['object', 'string']).columns
print('\n Categorical columns are ',categorical_columns)
print(df[categorical_columns].isnull().sum())
#(null values is not exists in categorical columns)
print('\n Locations size:',df['Location'].nunique())

# Encode categorical columns with one-hot encoding
df_encoded = pd.get_dummies(df, columns = categorical_columns, dtype = 'int')
print(df_encoded.columns.values)
#print(df.columns.values)

# Features scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(df_encoded), columns = df_encoded.columns, index = df_encoded.index)
#print(X.head())
# Final check
print('\n Final Shape of X :',X.shape)
print('\n Final Columns of X :',X.columns.values)
print('\n Final Data Types of X :',X.dtypes)
print('\n Final check for null values in X :',X.isnull().sum())
print('\n Final check for duplicates values in X :',X.duplicated().sum())