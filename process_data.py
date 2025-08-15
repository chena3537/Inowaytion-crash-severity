import pandas as pd
from src.features.build_features import str_to_dtype, landuse_preprocess, nsw_preprocess, lga


def main():
    crash_data_filepath = "data/raw/nsw_road_crash_data_2019-2023_crash.csv"
    landuse_data_filepath = "data/raw/syd_landuse_test.csv"
    lgas = ['Sydney']

    df = pd.read_csv(crash_data_filepath)
    landuse_df = (pd.read_csv(landuse_data_filepath))
    str_to_dtype(landuse_df, "landuse")

    landuse_data = landuse_preprocess(landuse_df)
    crash_data = nsw_preprocess(lga(df, lgas))
    data = pd.merge(crash_data, landuse_data, 'left', on='Crash ID')
    data.to_csv("data/processed/sydlga_crash_severity_processed.csv")

if __name__ == "__main__":
    main()