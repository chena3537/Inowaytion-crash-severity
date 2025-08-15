from src.visualization.network_map import load_hourly, create_hourly_map
from osmnx import geocode

def main():
    place_name = "Sydney, NSW, Australia"
    center_point = geocode("Town Hall, Sydney, Australia")
    hourly_preds = load_hourly('data/incidents/hourly_predictions_real.pkl')
    print(hourly_preds)
    create_hourly_map(place_name, center_point, hourly_preds, save_as= "maps/syd_hourly_predictions_real",load_graph="data/graphs/sydney_road_graph.pkl")

if __name__ == "__main__":
    main()