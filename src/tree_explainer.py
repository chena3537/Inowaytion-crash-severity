import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def explainTree(model, X_test, feature_names, crash_ids, topk=5, save=False, file_name="sample_risk_factors.csv"):
    rows = []
    encoder = LabelEncoder()
    y_pred = model.predict(X_test)
    encoder.fit(y_pred)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    for i in range(len(shap_values)):
        class_shaps = []
        for j in range(len(shap_values[0][0])):
            shap_for_class = shap_values[i,:,j]
            feature_ranks = sorted(list(zip(feature_names, shap_for_class)), key=lambda x: x[1], reverse=True)
            top_features, top_shap = zip(*feature_ranks[:topk])
            class_shaps.append((top_features, top_shap))

        rows.append({
        "Crash ID": crash_ids[i],
        "Top_factors_Fatal": list(class_shaps[0][0]),
        "SHAP_values_Fatal": [float(x) for x in class_shaps[0][1]],
        "Top_factors_Injury": list(class_shaps[1][0]),
        "SHAP_values_Injury": [float(x) for x in class_shaps[1][1]],
        "Top_factors_Non-casualty (towaway)": list(class_shaps[2][0]),
        "SHAP_values_Non-casualty (towaway)": [float(x) for x in class_shaps[2][1]]
    })
    df = pd.DataFrame(rows)
    if save:
        df.to_csv(f"Model Reports/{file_name}")

    return df
    