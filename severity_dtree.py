import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import calendar
import seaborn as sns
import os
import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE




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

def resample(X,y):
    X.fillna(0, inplace=True)
    return SMOTE().fit_resample(X,y)

def train_model(X_train, y_train, smote=False, **kwargs):
    model = tree.DecisionTreeClassifier(random_state=0, class_weight='balanced', **kwargs)

    if smote:
        X_train, y_train = resample(X_train, y_train)

    model = model.fit(X_train, y_train) 
    return model

def evaluate_model(model, X_test, y_test, feature_names, class_names, n_feature_importances = 10, save = False, output_dir='Model Evaluations', model_dir=None):
    y_pred = model.predict(X_test)

    if model_dir == None:
        model_id = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        model_dir = f'dtree_eval_{model_id}'
    model_dir_path = f'{output_dir}/{model_dir}'
    os.makedirs(model_dir_path, exist_ok=True)

    report = classification_report(y_test, y_pred)
    print("Model Hyperparameters:")
    print(model.get_params())
    print("\nClassification Report:")
    print(report)
    
    if save:
        with open(f'{model_dir_path}/evaluation_report.txt', 'w') as f:
            f.write("Model Hyperparameters:\n")
            f.write(str(model.get_params()))
            f.write("\n\nClassification Report:\n")
            f.write(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    if save:
        plt.savefig(f'{model_dir_path}/confusion_matrix.png')

    plt.figure(2)
    tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=class_names)
    plt.title('Decision Tree Structure')

    if save:
        plt.savefig(f'{model_dir_path}/decision_tree.png')
    
    plt.figure(3)
    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
    feature_importance.sort_values(ascending=False)[:n_feature_importances].plot(kind="bar")
    plt.title('Feature Importance')
    
    if save:
        plt.savefig(f'{model_dir_path}/feature_importance.png')

    plt.show()

def hourly_pred(model, X_train, save_path="hourly_predictions"):
    hourly_test = X_train.copy()
    hourly_preds = {}
    for d in range(7):
        hourly_preds[d] = {}
        for h in range(0,23,2):
            hourly_preds[d][h] = {}
            hourly_data = X_train[(X_train['Day of week of crash'] == d) & (X_train['Two-hour interval start'] == h)]
            if len(hourly_data) > 0:
                y_pred = model.predict(hourly_data)
                coords = list(zip(hourly_data.Latitude, hourly_data.Longitude))
            else:
                y_pred, coords = [],[]
            hourly_preds[d][h]['coordinates'] = coords
            hourly_preds[d][h]['severity'] = y_pred
    
    with open(f'{save_path}.pkl', 'wb') as f:
        pickle.dump(hourly_preds, f)


    

def main():
    crash_data_filepath = "Datasets/nsw_road_crash_data_2019-2023_crash.csv"
    landuse_data_filepath = "Datasets/syd_landuse_test.csv"
    lgas = ['Sydney']


    y_col = 'Degree of crash'
    df = pd.read_csv(crash_data_filepath)
    landuse_df = (pd.read_csv(landuse_data_filepath))
    str_to_dtype(landuse_df, "landuse")

    landuse_data = landuse_preprocess(landuse_df)
    crash_data = nsw_preprocess(lga(df, lgas))
    data = pd.merge(crash_data, landuse_data, 'left', on='Crash ID')

    X,y = data.drop([y_col,'Crash ID'], axis=1), data[y_col]
    feature_names = X.columns
    class_names = sorted(y.unique())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train)
    hourly_pred(model, X_train, "hourly_predictions_real")
    #evaluate_model(model, X_test, y_test, feature_names=feature_names, class_names=class_names, save=True)
    
    
    

if __name__ == "__main__":
    main()