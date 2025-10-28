import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from src.models.severity_dtree import train_model, hourly_pred, evaluate_model
from src.tree_explainer import explainTree
from src.visualization.network_map import load_graph, nearest_segments



def main():
    y_col = 'Degree of crash'
    data = read_csv("data/processed/sydlga_crash_severity_processed.csv", index_col=0)
    unprocessed_data = read_csv("data/raw/nsw_road_crash_data_2019-2023_crash.csv")
    print(unprocessed_data.columns)
    time_df = unprocessed_data[['Crash ID', 'Month of crash', 'Day of week of crash', 'Two-hour intervals']]
    X,y = data.drop([y_col], axis=1), data[y_col]
    feature_names = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    crash_ids_test = X_test['Crash ID'].tolist()
    X_train.drop(['Crash ID'], axis=1, inplace=True)
    X_test.drop(['Crash ID'], axis=1, inplace=True)
    model = train_model(X_train, y_train, max_depth=5)

    shap_values = explainTree(model, X_test.reset_index(drop=True), feature_names=feature_names, crash_ids=crash_ids_test, topk=5, save=False)

    graph = load_graph("data/graphs/sydney_road_graph.pkl")
    crash_segments = nearest_segments(graph, list(zip(X_test['Latitude'], X_test['Longitude'])))
    start_coords = [(graph.nodes[edge[0]]['y'], graph.nodes[edge[0]]['x']) for edge in crash_segments]
    end_coords = [(graph.nodes[edge[1]]['y'], graph.nodes[edge[1]]['x']) for edge in crash_segments]
    
    segments_df = pd.DataFrame({
        "Crash ID": crash_ids_test,
        "OSM_segment_ID": crash_segments,
        "Start_coordinates": start_coords,
        "End_coordinates": end_coords
    })
    segments_df = pd.merge(segments_df, time_df, "left", "Crash ID")

    segment_risk_factors = pd.merge(segments_df, shap_values, "left", "Crash ID")

    segment_risk_factors.to_csv("Model Reports/sample_segment_risk_factors.csv")

    

    #hourly_pred(model, X_train, "hourly_predictions_real")
    #evaluate_model(model, X_test, y_test, feature_names=feature_names, class_names=class_names, save=True)


if __name__ == "__main__":
    main()