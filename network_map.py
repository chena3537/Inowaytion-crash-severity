import folium
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import calendar



def load_hourly(file_path):
    with open(file_path, 'rb') as f:
        hourly_preds = pickle.load(f)

    return hourly_preds

def road_network_from_place(place):
    return ox.graph_from_place(place, network_type='drive')

def map_color(pred, colors=['red', 'orange', 'yellow'], class_names=['Fatal','Injury','Non-casualty (towaway)']):
    mapping = dict(zip(class_names, colors))
    return [mapping[i] for i in pred]

def nearest_segments(graph, coords):
    X,Y= zip(*coords)
    return ox.distance.nearest_edges(graph, list(Y), list(X))

def create_hourly_map(place_name, center_point, hourly_preds, colors=['red', 'orange', 'yellow'], class_names=['Fatal','Injury','Non-casualty (towaway)'], save_as = 'hourly_crash_severity_map', load_graph=None):
    if load_graph != None:
        with open(load_graph, "rb") as f:
            G = pickle.load(f)
    else:
        G = ox.graph_from_place(place_name, network_type='drive')

    m = folium.Map(location=[center_point[0], center_point[1]], zoom_start=10)
    crash_segments = nearest_segments(G, hourly_preds['coordinates'])
    
    
    for d in range(7):
        for h in range(0,23,2):
            layer = folium.FeatureGroup(name=f'Prediction - {calendar.day_name[d]} - {h}:00-{h+1}:59', show=False)
            colors = map_color(hourly_preds[d][h], colors=colors, class_names=class_names)
            for edge, color in zip(crash_segments, colors):
                edge_data = G.edges[*edge]
                geom = edge_data.get('geometry')
                if geom and geom.geom_type == 'LineString':
                    folium.PolyLine(locations=[(lat, lon) for lon, lat in geom.coords], color=color,weight=4).add_to(layer)
            layer.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    m.save("sydney_hourly_crash_severity.html")
    




    

def main():
    place_name = "Sydney, NSW, Australia"
    center_point = ox.geocode("Town Hall, Sydney, Australia")
    
    hourly_preds = load_hourly('hourly_predictions.pkl')
    print(hourly_preds)
    
    #create_hourly_map(place_name, center_point, hourly_preds, load_graph="sydney_road_graph.pkl")

if __name__ == "__main__":
    main()