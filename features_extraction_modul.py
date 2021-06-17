import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from tqdm import tqdm



def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)


def extract_features(train_data):
    seg_lenght = 150000
    total_samples = int(np.floor((train_data.shape[0])/ seg_lenght))
    print(total_samples)
    #an empty dataframe holding our feature values
    x_train = pd.DataFrame(index = range(total_samples), dtype = np.float64)
    #an empty dataframe holding our target labels
    y_train = pd.DataFrame(index = range(total_samples), columns = ['time_to_failure'], dtype = np.float64)
    
    
    
    for value in tqdm(range(total_samples)):
        sample = train_data.iloc[value* seg_lenght : value* seg_lenght + seg_lenght]
        x = pd.Series(sample['acoustic_data'].values)
        y = sample['time_to_failure'].values[-1]
        y_train.loc[value, 'time_to_failure'] = y
    
        x_train.loc[value, 'AVERAGE'] = x.mean()
        x_train.loc[value, 'STD'] = x.std()
        x_train.loc[value, 'MAX'] = x.max()
        x_train.loc[value, 'MIN'] = x.min() 
        x_train.loc[value, 'SUM'] = x.sum()
    
        x_train.loc[value, 'MEAN_CHANGE_ABS'] = np.mean(np.diff(x))
        x_train.loc[value, 'MEAN_CHANGE_RATE'] = calc_change_rate(x)
    
        x_train.loc[value, 'MAX_TO_MIN'] = x.max() / np.abs(x.min())
        x_train.loc[value, 'MAX_TO_MIN_DIFF'] = x.max() - np.abs(x.min())
        x_train.loc[value, 'COUNT_BIG'] = len(x[np.abs(x) > 500])
    
        x_train.loc[value, 'AVERAGE_FIRST_10000'] = x[:10000].mean()
        x_train.loc[value, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()
        x_train.loc[value, 'AVERAGE_FIRST_50000'] = x[:50000].mean()
        x_train.loc[value, 'AVERAGE_LAST_50000'] = x[-50000:].mean()
    
        x_train.loc[value, 'STD_FIRST_10000'] = x[:10000].std()
        x_train.loc[value, 'STD_LAST_10000']  =  x[-10000:].std()
        x_train.loc[value, 'STD_FIRST_50000'] = x[:50000].std()
        x_train.loc[value, 'STD_LAST_50000'] = x[-50000:].std()
    
        x_train.loc[value, 'ABS_AVERAGE'] = np.abs(x).mean()
        x_train.loc[value, 'ABS_STD'] = np.abs(x).std()
        x_train.loc[value, 'ABS_MAX'] = np.abs(x).max()
        x_train.loc[value, 'ABS_MIN'] = np.abs(x).min()
    
        x_train.loc[value, '10Q'] = np.percentile(x, 0.10)
        x_train.loc[value, '25Q'] = np.percentile(x, 0.25)
        x_train.loc[value, '50Q'] = np.percentile(x, 0.50)
        x_train.loc[value, '75Q'] = np.percentile(x, 0.75)
        x_train.loc[value, '90Q'] = np.percentile(x, 0.90)
    
        x_train.loc[value, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))
        x_train.loc[value, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))
        x_train.loc[value, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))
        x_train.loc[value, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))
        x_train.loc[value, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))
        x_train.loc[value, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))
        
        x_train.loc[value, 'KURTOSIS'] = x.kurtosis()
        x_train.loc[value, 'SKEW'] = x.skew()
        x_train.loc[value, 'MEDIAN'] = x.median()
        
        x_train.loc[value, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()
        x_train.loc[value, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()
        
        for windows in [10, 100, 1000]:
            x_roll_std = x.rolling(windows).std().dropna().values
            x_roll_mean = x.rolling(windows).mean().dropna().values
            
            x_train.loc[value, 'AVG_ROLL_STD' + str(windows)] = x_roll_std.mean()
            x_train.loc[value, 'STD_ROLL_STD' + str(windows)] = x_roll_std.std()
            x_train.loc[value, 'MAX_ROLL_STD' + str(windows)] = x_roll_std.max()
            x_train.loc[value, 'MIN_ROLL_STD' + str(windows)] = x_roll_std.min()
            x_train.loc[value, '1Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.01)
            x_train.loc[value, '5Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.05)
            x_train.loc[value, '95Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.95)
            x_train.loc[value, '99Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.99)
            x_train.loc[value, 'AV_CHANGE_ABS_ROLL_STD' + str(windows)] = np.mean(np.diff(x_roll_std))
            x_train.loc[value, 'ABS_MAX_ROLL_STD' + str(windows)] = np.abs(x_roll_std).max()
            
            x_train.loc[value, 'AVG_ROLL_MEAN' + str(windows)] = x_roll_mean.mean()
            x_train.loc[value, 'STD_ROLL_MEAN' + str(windows)] = x_roll_mean.std()
            x_train.loc[value, 'MAX_ROLL_MEAN' + str(windows)] = x_roll_mean.max()
            x_train.loc[value, 'MIN_ROLL_MEAN' + str(windows)] = x_roll_mean.min()
            x_train.loc[value, '1Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            x_train.loc[value, '5Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            x_train.loc[value, '95Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            x_train.loc[value, '99Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            x_train.loc[value, 'AV_CHANGE_ABS_ROLL_MEAN' + str(windows)] = np.mean(np.diff(x_roll_mean))
            x_train.loc[value, 'ABS_MAX_ROLL_MEAN' + str(windows)] = np.abs(x_roll_mean).max()

    return x_train, y_train
            
def extract_features_with_shift(train_data, shift = 25000, t = 9):
    seg_lenght = 150000
    shifts_in_seg_lenght = 150000/shift
    
    x_array = []
    y_array = []
    
    for i in np.arange(0,  int(shifts_in_seg_lenght)):#int(shifts_in_seg_lenght)):
        shifted_train_data = train_data[i*shift :]
        
        shifted_train_data = shifted_train_data[shifted_train_data["time_to_failure"] > t]
        
        i_x, i_y = extract_features(shifted_train_data)
        x_array.append(i_x)
        y_array.append(i_y)
        
    return pd.concat(x_array, ignore_index = True), pd.concat(y_array, ignore_index = True)

    
def test_extract_features(train_data):
    seg_lenght = 150000
    total_samples = int(np.floor((train_data.shape[0])/ seg_lenght))
    print(total_samples)
    #an empty dataframe holding our feature values
    x_train = pd.DataFrame(index = range(total_samples), dtype = np.float64)
    #an empty dataframe holding our target labels
    
    
    
    for value in tqdm(range(total_samples)):
        sample = train_data.iloc[value* seg_lenght : value* seg_lenght + seg_lenght]
        x = pd.Series(sample['acoustic_data'].values)
    
        x_train.loc[value, 'AVERAGE'] = x.mean()
        x_train.loc[value, 'STD'] = x.std()
        x_train.loc[value, 'MAX'] = x.max()
        x_train.loc[value, 'MIN'] = x.min() 
        x_train.loc[value, 'SUM'] = x.sum()
    
        x_train.loc[value, 'MEAN_CHANGE_ABS'] = np.mean(np.diff(x))
        x_train.loc[value, 'MEAN_CHANGE_RATE'] = calc_change_rate(x)
    
        x_train.loc[value, 'MAX_TO_MIN'] = x.max() / np.abs(x.min())
        x_train.loc[value, 'MAX_TO_MIN_DIFF'] = x.max() - np.abs(x.min())
        x_train.loc[value, 'COUNT_BIG'] = len(x[np.abs(x) > 500])
    
        x_train.loc[value, 'AVERAGE_FIRST_10000'] = x[:10000].mean()
        x_train.loc[value, 'AVERAGE_LAST_10000']  =  x[-10000:].mean()
        x_train.loc[value, 'AVERAGE_FIRST_50000'] = x[:50000].mean()
        x_train.loc[value, 'AVERAGE_LAST_50000'] = x[-50000:].mean()
    
        x_train.loc[value, 'STD_FIRST_10000'] = x[:10000].std()
        x_train.loc[value, 'STD_LAST_10000']  =  x[-10000:].std()
        x_train.loc[value, 'STD_FIRST_50000'] = x[:50000].std()
        x_train.loc[value, 'STD_LAST_50000'] = x[-50000:].std()
    
        x_train.loc[value, 'ABS_AVERAGE'] = np.abs(x).mean()
        x_train.loc[value, 'ABS_STD'] = np.abs(x).std()
        x_train.loc[value, 'ABS_MAX'] = np.abs(x).max()
        x_train.loc[value, 'ABS_MIN'] = np.abs(x).min()
    
        x_train.loc[value, '10Q'] = np.percentile(x, 0.10)
        x_train.loc[value, '25Q'] = np.percentile(x, 0.25)
        x_train.loc[value, '50Q'] = np.percentile(x, 0.50)
        x_train.loc[value, '75Q'] = np.percentile(x, 0.75)
        x_train.loc[value, '90Q'] = np.percentile(x, 0.90)
    
        x_train.loc[value, 'ABS_1Q'] = np.percentile(x, np.abs(0.01))
        x_train.loc[value, 'ABS_5Q'] = np.percentile(x, np.abs(0.05))
        x_train.loc[value, 'ABS_30Q'] = np.percentile(x, np.abs(0.30))
        x_train.loc[value, 'ABS_60Q'] = np.percentile(x, np.abs(0.60))
        x_train.loc[value, 'ABS_95Q'] = np.percentile(x, np.abs(0.95))
        x_train.loc[value, 'ABS_99Q'] = np.percentile(x, np.abs(0.99))
        
        x_train.loc[value, 'KURTOSIS'] = x.kurtosis()
        x_train.loc[value, 'SKEW'] = x.skew()
        x_train.loc[value, 'MEDIAN'] = x.median()
        
        x_train.loc[value, 'HILBERT_MEAN'] = np.abs(hilbert(x)).mean()
        x_train.loc[value, 'HANN_WINDOW_MEAN'] = (convolve(x, hann(150), mode = 'same') / sum(hann(150))).mean()
        
        for windows in [10, 100, 1000]:
            x_roll_std = x.rolling(windows).std().dropna().values
            x_roll_mean = x.rolling(windows).mean().dropna().values
            
            x_train.loc[value, 'AVG_ROLL_STD' + str(windows)] = x_roll_std.mean()
            x_train.loc[value, 'STD_ROLL_STD' + str(windows)] = x_roll_std.std()
            x_train.loc[value, 'MAX_ROLL_STD' + str(windows)] = x_roll_std.max()
            x_train.loc[value, 'MIN_ROLL_STD' + str(windows)] = x_roll_std.min()
            x_train.loc[value, '1Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.01)
            x_train.loc[value, '5Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.05)
            x_train.loc[value, '95Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.95)
            x_train.loc[value, '99Q_ROLL_STD' + str(windows)] = np.quantile(x_roll_std, 0.99)
            x_train.loc[value, 'AV_CHANGE_ABS_ROLL_STD' + str(windows)] = np.mean(np.diff(x_roll_std))
            x_train.loc[value, 'ABS_MAX_ROLL_STD' + str(windows)] = np.abs(x_roll_std).max()
            
            x_train.loc[value, 'AVG_ROLL_MEAN' + str(windows)] = x_roll_mean.mean()
            x_train.loc[value, 'STD_ROLL_MEAN' + str(windows)] = x_roll_mean.std()
            x_train.loc[value, 'MAX_ROLL_MEAN' + str(windows)] = x_roll_mean.max()
            x_train.loc[value, 'MIN_ROLL_MEAN' + str(windows)] = x_roll_mean.min()
            x_train.loc[value, '1Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            x_train.loc[value, '5Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            x_train.loc[value, '95Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            x_train.loc[value, '99Q_ROLL_MEAN' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            x_train.loc[value, 'AV_CHANGE_ABS_ROLL_MEAN' + str(windows)] = np.mean(np.diff(x_roll_mean))
            x_train.loc[value, 'ABS_MAX_ROLL_MEAN' + str(windows)] = np.abs(x_roll_mean).max()

    return x_train 
            
