import pandas as pd
import numpy as np
import ast
import calendar




def load_data(filepath):
    return pd.read_csv(filepath)

def str_to_dtype(df,col):
    df[col] = df[col].apply(ast.literal_eval)

def lga(df, lga_list):
     return df[df['LGA'].isin(lga_list)]

def merge_data(col, df1, df2, *args):
    merged_df = pd.merge(df1, df2, "left", col)
    for arg in args:
        merged_df = pd.merge(merged_df, arg, "left", col)
    return merged_df

def nsw_convert_hours(df, col):
    df[col].replace('Midnight', '23:59', regex = True, inplace=True)
    df[col].replace('Unknown', np.nan, inplace=True)
    
    df[['Two-hour interval start', 'Two-hour interval end']] = df[col].str.split(' - ', expand = True)
    df['Two-hour interval start'] = pd.to_datetime(df['Two-hour interval start'], format='%H:%M', errors = 'ignore').dt.hour.astype(int, errors='ignore')
    df['Two-hour interval end'] = pd.to_datetime(df['Two-hour interval end'], format='%H:%M', errors = 'ignore').dt.hour + 1
    df.drop([col], axis=1, inplace=True)

def convert_days_num(df, col):
    day_map = {day: i for i, day in enumerate(calendar.day_name)}
    df[col] = df[col].map(day_map)

def convert_months_num(df, col):
    df[col] = pd.to_datetime(df[col], format='%B').dt.month
    
def nsw_convert_to_num(df):
    convert_months_num(df, 'Month of crash')
    convert_days_num(df, 'Day of week of crash')
    nsw_convert_hours(df, 'Two-hour intervals')
    df['Speed limit'].replace('Unknown', np.nan, inplace=True)
    df['Speed limit'] = df['Speed limit'].str.replace(' km/h', '', regex=False).astype(int, errors = 'ignore')
    df['Street lighting'].replace('Unknown / not stated', np.nan, inplace=True)
    df['Signals operation'].replace('Unknown / not stated', np.nan, inplace=True)
    
def create_onehot(df, cols):
    oh_encoded = pd.get_dummies(df, prefix=cols, columns=cols, dtype=int)
    return oh_encoded

def nsw_preprocess(df):
    cols = ['Crash ID','Degree of crash', 'Year of crash', 'Month of crash', 'Day of week of crash', 'Two-hour intervals', 
            'Street type', 'Distance', 'Direction', 'School zone location', 'School zone active', 'Type of location', 'Latitude', 'Longitude', 
            'Urbanisation', 'Alignment', 'Primary permanent feature', 'Primary hazardous feature' ,'Street lighting', 'Road surface', 'Surface condition', 
            'Weather', 'Natural lighting', 'Signals operation', 'Other traffic control', 'Speed limit', 'Road classification (admin)', 'First impact type',
            'Key TU type', 'Other TU type', 'No. of traffic units involved']
    oh_cols = ['Street type', 'Direction', 'School zone location', 'School zone active', 'Type of location', 'Urbanisation', 'Alignment', 'Primary permanent feature', 'Primary hazardous feature' ,'Street lighting', 'Road surface', 'Surface condition', 
            'Weather', 'Natural lighting', 'Signals operation', 'Other traffic control', 'Road classification (admin)', 'First impact type',
            'Key TU type', 'Other TU type']
    processed_df = df[cols]
    nsw_convert_to_num(processed_df)
    oh_encoded = create_onehot(processed_df, oh_cols)
    return oh_encoded

def landuse_preprocess(df):
    df_exploded = df.explode("landuse")
    oh_exploded = pd.get_dummies(df_exploded["landuse"], dtype=int)
    oh_encoded = oh_exploded.groupby(oh_exploded.index).any().astype(int)
    return df['Crash ID'].to_frame().join(oh_encoded)