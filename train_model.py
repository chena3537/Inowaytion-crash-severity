from pandas import read_csv
from sklearn.model_selection import train_test_split
from src.models.severity_dtree import train_model, hourly_pred, evaluate_model



def main():
    y_col = 'Degree of crash'
    data = read_csv("data/processed/sydlga_crash_severity_processed.csv")
    X,y = data.drop([y_col,'Crash ID'], axis=1), data[y_col]
    feature_names = X.columns
    class_names = sorted(y.unique())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = train_model(X_train, y_train, max_depth=5)
    #hourly_pred(model, X_train, "hourly_predictions_real")
    evaluate_model(model, X_test, y_test, feature_names=feature_names, class_names=class_names, save=True)


if __name__ == "__main__":
    main()