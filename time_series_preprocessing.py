# Data reading in Dataframe format and data preprocessing
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Linear algebra operations
import numpy as np 

# Machine learning models and preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Epiweek
from epiweeks import Week, Year

# Date
from datetime import date as convert_to_date

import warnings
warnings.filterwarnings('ignore')


""" Get Epiweek as int """
def epiweek_from_date(image_date):
    date = image_date.split('-')
    
    # Get year as int
    year = ''.join(filter(str.isdigit, date[0]))
    year = int(year)
    
    # Get month as int
    month = ''.join(filter(str.isdigit, date[1]))
    month = int(month)
    
    # Get day as int
    day = ''.join(filter(str.isdigit, date[2]))
    day = int(day)
    
    # Get epiweek:
    date = convert_to_date(year, month, day)
    epiweek = str(Week.fromdate(date))
    epiweek = int(epiweek)
    
    return epiweek


""" Get epiweek as column name """
def get_epiweek(name):
    
    # Get week
    week = name.split('/')[1]
    week = week.replace('w','')
    week = int(week)
    
    # Year
    year = name.split('/')[0]
    year = int(year)
    
    epiweek = Week(year, week)
    
    epiweek = str(epiweek)
    epiweek = int(epiweek)

    return epiweek



def train_test_split(df, train_percentage = 80):
    # We need a sequence so we can't split randomly
    # To divide into Train and test we have to calculate the train percentage of the dataset:
    size = df.shape[0]
    split = int(size*(train_percentage/100))
    
    """ Train """
    # We will train with 1st percentage % of data and test with the rest
    train_df = df.iloc[:split,:] ## percentage % train
    
    """ Test """
    test_df = df.iloc[split:,:] # 100 - percentage % test
    
    print(f'The train shape is: {train_df.shape}')
    print(f'The test shape is: {test_df.shape}')
    
    return train_df, test_df


""" Train-Test Split"""
def train_test_split(df, train_percentage = 80):
    # We need a sequence so we can't split randomly
    # To divide into Train and test we have to calculate the train percentage of the dataset:
    size = df.shape[0]
    split = int(size*(train_percentage/100))
    
    """ Train """
    # We will train with 1st percentage % of data and test with the rest
    train_df = df.iloc[:split,:] ## percentage % train
    
    """ Test """
    test_df = df.iloc[split:,:] # 100 - percentage % test
    
    print(f'The train shape is: {train_df.shape}')
    print(f'The test shape is: {test_df.shape}')
    
    return train_df, test_df

""" Normalization"""
# Normalize train data and create the scaler
def normalize_train_features(df, feature_range=(-1, 1), scaler=True, describe=None):
    
    scalers = {}
    # For each column in the dataframe
    for i, column in enumerate(df.columns):
        if not scaler:
            if (i == len(df.columns) - 1):
                continue
        
        # Get values of the column
        values = df[column].values.reshape(-1,1)
        if not(feature_range):
            # Generate a new scaler
            scaler = StandardScaler()
        else:
            # Generate a new scaler
            scaler = MinMaxScaler(feature_range=feature_range)
        # Fit the scaler just for that column
        scaled_column = scaler.fit_transform(values)
        # Add the scaled column to the dataframe
        scaled_column = np.reshape(scaled_column, len(scaled_column))
        df[column] = scaled_column
        
        # Save the scaler of the column
        scalers['scaler_' + column] = scaler
    if describe:
        print(f' Min values are: ')
        print(df.min())
        print(f' Max values are: ')
        print(df.max())
        
    return df, scalers


""" If you want to use the same scaler used in train, you can use this function"""
def normalize_test_features(df, scalers=None, scaler=True, describe=None):
    
    if not scalers:
        raise TypeError("You should provide a list of scalers.")
        
    for i, column in enumerate(df.columns):
        
        if not scaler:
            if (i == len(df.columns) - 1):
                continue
        
        # Get values of the column
        values = df[column].values.reshape(-1,1)
        # Take the scaler of that column
        scaler = scalers['scaler_' + column]
        # Scale values
        scaled_column = scaler.transform(values)
        scaled_column = np.reshape(scaled_column,len(scaled_column))
        # Add the scaled values to the df
        df[column] = scaled_column
    if describe:
        print(f' Min values are: ')
        print(df.min())
        print(f' Max values are: ')
        print(df.max())
        
    return df 


# prepare data for time series
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, autoregressive=True):
    no_autoregressive = not(autoregressive)
    if no_autoregressive:
        n_in = n_in - 1
        
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        if no_autoregressive:
            cols.append(df.shift(i).iloc[:,:-1])
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
        else:
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

""" Features and Labels """
def features_labels_set(timeseries_data, original_df, autoregressive):
    
    """ Features """
    # We define the number of features as (Cases and media cloud)
    n_features = original_df.shape[1]

    # The features to train the model will be all except the values of the actual week 
    # We can't use other variables in week t because whe need to resample a a 3D Array
    if autoregressive:
        features_set = DataFrame(timeseries_data.values[:,:-n_features])
    else:
        features_set = DataFrame(timeseries_data.values[:,:-1])
    # Convert pandas data frame to np.array to reshape as 3D Array
    features_set = features_set.to_numpy()
    print(f'The shape of the features is {features_set.shape}')
    
    """ Labels """
    # We will use Covid cases in last week 
    labels_set = DataFrame(timeseries_data.values[:,-1])
    # Convert pandas data frame to np.array
    labels_set = labels_set.to_numpy()
    print(f'The shape of the labels is {labels_set.shape}')
    
    return features_set, labels_set, n_features


def reshape_tensor(train_X, test_X, n_features, days, autoregressive=True):
    print('The initial shapes are:')
    print(f'The train shape is {train_X.shape}')
    print(f'The test shape is {test_X.shape}')
    
    # reshape input to be 3D [samples, timesteps, features]
    if not(autoregressive):
        train_X = train_X.reshape((train_X.shape[0], days, n_features-1))
        test_X = test_X.reshape((test_X.shape[0], days, n_features-1))
    
    else:
        train_X = train_X.reshape((train_X.shape[0], days, n_features))
        test_X = test_X.reshape((test_X.shape[0], days, n_features))
    
    print('-----------------------')
    print('The Final shapes are:')
    print(f'The train shape is {train_X.shape}')
    print(f'The test shape is {test_X.shape}')
    
    return train_X, test_X

def preprocess_dataset_to_time_series(df, train_percentage = 80, feature_range = (-1, 1), T=3, autoregressive = False, normalize=True, reshape=True):
    """ Train-Test Split"""
    train_df, test_df = train_test_split(df, train_percentage = train_percentage)
    """ Normalization """
    if normalize:
        # Train:
        train_df, scalers = normalize_train_features(train_df, feature_range=feature_range)
        # Test:
        test_df = normalize_test_features(test_df, scalers=scalers)
    """ Generate Time Frame"""
    # Train:
    train = series_to_supervised(train_df, n_in=T, autoregressive=autoregressive)
    # Test:
    test = series_to_supervised(test_df, n_in=T, autoregressive=autoregressive)
    """ Features and Labels set"""
    # Train:
    train_X, train_y, n_features = features_labels_set(timeseries_data=train, original_df=df, autoregressive=autoregressive)
    # Test:
    test_X, test_y, n_features = features_labels_set(timeseries_data=test, original_df=df, autoregressive=autoregressive)
    """ Reshape """
    if reshape:
        # reshape input to be 3D [samples, timesteps, features]
        train_X, test_X = reshape_tensor(train_X, test_X, n_features, T, autoregressive)
    if normalize:
        return train_X, test_X, train_y, test_y, scalers
    else:
        return train_X, test_X, train_y, test_y