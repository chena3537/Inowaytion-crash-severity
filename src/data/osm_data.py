import osmnx as ox 
import geopandas as gpd 
import pandas as pd
import numpy as np
import sys

crash_data = "nsw_road_crash_data_2019-2023_crash.csv"
crash_df = pd.read_csv(crash_data)
LGA = "Sydney"

def lga_coords(crashdata, lga):
    lga_data = crashdata[crashdata['LGA'] == lga].reset_index()
    lga_data["coords"] = list(zip(lga_data.Latitude, lga_data.Longitude))
    return lga_data[["Crash ID", "coords"]]

def osm_features(data, tags):
    coords = data["coords"].tolist()
    feats = list(tags.keys())
    
    results = dict([(feat, []) for feat in feats])
    print(results)

    for index,coord in enumerate(coords):
        print(f'\rProgress: {index} / {len(coords)}', end='', flush=True)
        is_result = True
        try:
            response = ox.features.features_from_point(coord, tags=tags, dist=100)
        except ox._errors.InsufficientResponseError:
            is_result = False
        if is_result:
            for feat in feats:
                try:
                    result_data = list(response[feat].values)
                    results[feat].append(result_data)
                except KeyError:
                    results[feat].append([])
        else:
            for feat in feats:
                results[feat].append([])
    
    for feat in feats:
        data[feat] = results[feat]
    return data
 
    



def main():
    coord_data = lga_coords(crash_df, LGA)
    tags = {"landuse":True}
    df = osm_features(coord_data,tags)
    df.to_csv("syd_landuse_test.csv")

if __name__ == "__main__":
    main()