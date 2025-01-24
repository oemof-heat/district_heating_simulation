# -*- coding: utf-8 -*-

"""
Create a district heating network from OpenStreetMap data,
and perform a DHS Investment Optimisation.

Overview
--------

Part I: Get OSM data
Part II: Process the geometry for DHNx
Part III: Initialise the ThermalNetwork and perform the Optimisation
Part IV: Check the results

Contributors:
- Joris Zimmermann
- Johannes Röder
"""
import numpy as np
import osmnx as ox
from shapely import geometry
import matplotlib.pyplot as plt

import logging
from oemof.tools import logger

from dhnx.network import ThermalNetwork
from dhnx.input_output import load_invest_options
from dhnx.gistools.connect_points import process_geometry

logger.define_logging(
    screen_level=logging.INFO,
    logfile="dhnx.log"
)

# Part I: Get OSM data #############

# select the street types you want to consider as DHS routes
# see: https://wiki.openstreetmap.org/wiki/Key:highway
streets = dict({
    'highway': [
        'residential',
        'service',
        'unclassified',
    ]
})

# select the building types you want to import
# see: https://wiki.openstreetmap.org/wiki/Key:building
buildings = dict({
    'building': [
        'apartments',
        'commercial',
        'detached',
        'house',
        'industrial',
        'residential',
        'retail',
        'semidetached_house'
    ]
})

# Define a bounding box polygon from a list of lat/lon coordinates
bbox = [(9.1008896, 54.1954005),
        (9.1048374, 54.1961024),
        (9.1090996, 54.1906397),
        (9.1027474, 54.1895923),
        ]
polygon = geometry.Polygon(bbox)
graph = ox.graph_from_polygon(polygon, network_type='drive_service')
ox.plot_graph(graph)

gdf_poly_houses = ox.features_from_polygon(polygon, tags=buildings)
gdf_lines_streets = ox.features_from_polygon(polygon, tags=streets)

# Make sure that only polygon geometries are used
gdf_poly_houses = gdf_poly_houses[gdf_poly_houses['geometry'].apply(
    lambda x: isinstance(x, geometry.Polygon))]

# We need one (or more) buildings that we call "generators".
# Choose one among the buildings at random and move it to a new GeoDataFrame
np.random.seed(42)
id_generator = np.random.randint(len(gdf_poly_houses))
gdf_poly_gen = gdf_poly_houses.iloc[[id_generator]].copy()
gdf_poly_houses.drop(index=gdf_poly_houses.index[id_generator], inplace=True)

# The houses need a maximum thermal power. For this example, we set it
# to a random value between 10 and 50 kW for all houses
gdf_poly_houses['P_heat_max'] = \
    np.random.randint(10, 50, size=len(gdf_poly_houses))

# plot the given geometry
fig, ax = plt.subplots()
gdf_lines_streets.plot(ax=ax, color='blue')
gdf_poly_gen.plot(ax=ax, color='orange')
gdf_poly_houses.plot(ax=ax, color='green')
plt.title('Geometry before processing')
plt.show()

# # Optionally export the imported geometry (e.g. for QGIS)
# gdf_poly_houses.to_file('footprint_buildings.geojson', driver='GeoJSON')

# Part II: Process the geometry for DHNx #############

# # optionally you can skip Part I and load your own layer with geopandas, e.g.
# gdf_lines_streets = gpd.read_file('your_file.geojson')
# gdf_poly_gen = gpd.read_file('your_file.geojson')
# gdf_poly_houses = gpd.read_file('your_file.geojson')

# process the geometry
tn_input = process_geometry(
    lines=gdf_lines_streets,
    producers=gdf_poly_gen,
    consumers=gdf_poly_houses,
    method="boundary",  # select the method of how to connect the buildings
)

# plot output after processing the geometry
_, ax = plt.subplots()
tn_input['consumers'].plot(ax=ax, color='green')
tn_input['producers'].plot(ax=ax, color='red')
tn_input['pipes'].plot(ax=ax, color='blue')
tn_input['forks'].plot(ax=ax, color='grey')
plt.title('Geometry after processing')
plt.show()

# # optionally export the geodataframes and load it into qgis, arcgis whatever
# # for checking the results of the geometry processing
# path_geo = 'qgis'
# for key, val in tn_input.items():
#     val.to_file(os.path.join(path_geo, key + '.geojson'), driver='GeoJSON')


# Part III: Initialise the ThermalNetwork and perform the Optimisation #######

# initialize a ThermalNetwork
network = ThermalNetwork()

# add the pipes, forks, consumer, and producers to the ThermalNetwork
for k, v in tn_input.items():
    network.components[k] = v

# check if ThermalNetwork is consistent
network.is_consistent()

# load the specification of the oemof-solph components
invest_opt = load_invest_options('invest_data')


# optionally, define some settings for the solver. Especially increasing the
# solution tolerance with 'ratioGap' or setting a maximum runtime in 'seconds'
# helps if large networks take too long to solve
settings = dict(solver='cbc',
                solve_kw={
                    'tee': False,  # print solver output
                },
                solver_cmdline_options={
                    # 'allowableGap': 1e-5,  # (absolute gap) default: 1e-10
                    # 'ratioGap': 0.2,  # (0.2 = 20% gap) default: 0
                    # 'seconds': 60 * 1,  # (maximum runtime) default: 1e+100
                },
                )

# perform the investment optimisation
network.optimize_investment(invest_options=invest_opt, **settings)


# Part IV: Check the results #############

# get results
results_edges = network.results.optimization['components']['pipes']
# print(results_edges[['from_node', 'to_node', 'hp_type', 'capacity',
#                      'direction', 'costs', 'losses']])

print(results_edges[['costs']].sum())
print('Objective value: ', network.results.optimization['oemof_meta']['objective'])
# (The costs of the objective value and the investment costs of the DHS
# pipelines are the same, since no additional costs (e.g. for energy sources)
# are considered in this example.)

# add the investment results to the geoDataFrame
gdf_pipes = network.components['pipes']
gdf_pipes = gdf_pipes.join(results_edges, rsuffix='results_')

# plot output after processing the geometry
_, ax = plt.subplots()
network.components['consumers'].plot(ax=ax, color='green')
network.components['producers'].plot(ax=ax, color='red')
network.components['forks'].plot(ax=ax, color='grey')
gdf_pipes[gdf_pipes['capacity'] > 0].plot(ax=ax, color='blue')
plt.title('Invested pipelines')
plt.show()
