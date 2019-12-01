import geopandas as gpd
import pandas as pd

from shapely.ops import nearest_points
from shapely.geometry import LineString


def connect_points_to_network(points, nodes, edges):
    r"""

    Parameter
    ---------
    points :

    nodes : geopandas.GeoDataFrame
        Nodes of the network

    edges : geopandas.GeoDataFrame
        Edges of the network

    Returns
    -------
    points :
    nodes :
    edges :
    """
    edges_united = edges.unary_union

    len_nodes = len(nodes)
    len_points = len(points)

    # assign ids to new points
    n_points = []
    n_nearest_points = []
    n_edges = []

    for i, point in enumerate(points.geometry):
        id_nearest_point = len_nodes + i

        id_point = len_nodes + len_points + i

        nearest_point = nearest_points(edges_united, point)[0]

        n_points.append([id_point, point.x, point.y, point])

        n_nearest_points.append([id_nearest_point, nearest_point.x, nearest_point.y, nearest_point])

        n_edges.append([id_point, id_nearest_point, LineString([point, nearest_point])])

    n_points = gpd.GeoDataFrame(
        n_points,
        columns=['index', 'x', 'y', 'geometry']).set_index('index')

    n_nearest_points = gpd.GeoDataFrame(
        n_nearest_points,
        columns=['index', 'x', 'y', 'geometry']).set_index('index')

    n_edges = gpd.GeoDataFrame(n_edges, columns=['u', 'v', 'geometry'])

    joined_nodes = pd.concat([nodes, n_nearest_points], sort=True)
    joined_edges = pd.concat([edges, n_edges], sort=True)

    return n_points, joined_nodes, joined_edges