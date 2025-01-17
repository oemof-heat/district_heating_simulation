# -*- coding: utf-8 -*-

"""
This module holds functions for processing the geometry for setting up
the geometry of a ThermalNetwork based on a street geometry and a table of
buildings.

This file is part of project dhnx (). It's copyrighted
by the contributors recorded in the version control history of the file,
available from its original location: https://github.com/oemof/DHNx

This module is not fully tested yet, so use it with care.

SPDX-License-Identifier: MIT
"""
try:
    import geopandas as gpd

except ImportError:
    print("Need to install geopandas to process geometry data.")

try:
    from shapely.geometry import LineString
    from shapely.geometry import MultiPoint
    from shapely.geometry import Point
    from shapely.ops import nearest_points
    from shapely.ops import unary_union
except ImportError:
    print("Need to install shapely to process geometry.")

import logging

import numpy as np
import pandas as pd

from . import geometry_operations as go

logger = logging.getLogger(__name__)  # Create a logger for this module


def line_of_point(point, gdf_lines):
    """Gets index of geometry of a GeoDataFrame, a point is located next to,
      with a distance lower than 1e-8.

    Parameters
    ----------
    point : shapely.geometry.Point
    gdf_lines : geopandas.GeoDataFrame

    Returns
    -------
    int, float or str : Index of GeoDataFrame or Warning, if no geometry found.
    """
    ind = None

    for k, l in gdf_lines.iterrows():

        if l['geometry'].distance(point) < 1e-8:
            ind = k

    if ind is None:
        return Warning('No line found which has point on it!')
    return ind


def point_to_array(point):
    """Returns the coordinates of a point as numpy.array

    Parameters
    ----------
    point : shapely.geometry.Point

    Returns
    -------
    numpy.array()
    """

    return np.array([point.x, point.y])


def calc_lot_foot(line, point):
    """
    Calculates the lot foot point.

    Parameters
    ----------
    line : shapely.geometry.LineString
    point : shapely.geometry.Point

    Returns
    -------
    shapely.geometry.Point
    """
    s_1 = Point(line.coords[0])
    s_2 = Point(line.coords[-1])

    g_1 = point_to_array(s_1)  # end point 1 of line
    g_2 = point_to_array(s_2)  # end point 2 of line

    x_1 = point_to_array(point)

    # calculate lotfusspunkt
    u = g_2 - g_1  # vector of direction
    n = np.array([u[1], -u[0]])  # normal vector of line
    x_0 = g_1  # point on line

    y = x_1 - (np.dot((x_1 - x_0), n) / np.dot(n, n)) * n
    lot_foot_point = Point(y[0], y[1])

    # # alternative generation via intersection
    # # (=> intersections point is not exaclty on lines as well)
    # y = x_1 - 2*(np.dot((x_1 - x_0), n)/np.dot(n, n)) * n
    # lot_line = LineString([(y[0], y[1]), (x_1[0], x_1[1])])
    # lot_foot_point = lot_line.intersection(line)

    return lot_foot_point


def create_object_connections(points, lines, tol_distance=1):
    """Connects points to a line network.

    Generally, the nearest point of the next line is used as connection the point.
    Depending on the geometry, there are 3 options, the connection is created:

    - nearest point is line ending => the connection line starts from this line ending

    - nearest point is on the next line:

      a) line endings are outside the tolerance => line is split and the nearest point
      is used as connection point

      b) line endings are within the tolerance distance => the next line ending is
      used as connection point

    The tolerance distance avoids the generation of short line elements.
    This is for example the case if two buildings are directly opposite of the street.
    Using simply the nearest point method could result in very short lines.


    Parameters
    ----------
    points : geopandas.GeoDataFrame
        Points which should be connected to the line. GeoDataFrame with Points as geometry.
    lines : geopandas.GeoDataFrame
        The line-network to which the Points should be connected. The line geometry needs to
        consists of simple lines based on one starting and one ending point. LineStrings
        which contain more than 2 points are not allowed.
    tol_distance : float
        Tolerance distance for choosing the end of the line instead of the nearest point.

    Returns
    -------
    geopandas.GeoDataFrame : The newly created connection lines
    geopandas.GeoDataFrame : The updated lines (some lines are split.
        All lines should only touch at the line endings.

    """
    # check linestrings
    for _, c in lines.iterrows():
        if len(c['geometry'].coords) > 2:
            raise ValueError("The Linestrings must consists of simple lines,"
                             " with only two coordinates!")

    # Pepare merging all the street lines
    all_lines = lines['geometry']

    # There seems to be a conflict between shapely and pygeos,
    # which both use 'geos' internally, if both are installed.
    # This can cause
    # 'OSError exception: access violation reading 0xFFFFFFFFFFFFFFFF'.
    #
    # With shapely 1.8.0 and pygeos 0.12.0 it was observed that
    # this sometimes even fails without error. In such a case
    # mergedlines might only contain a single LineString (one street
    # segment) instead of a MultiLineString (the combined network
    # of all street segments). This completely messes up the
    # following nearest_points().
    #
    # Wrapping the argument in 'list()' seems to be a valid workaround.
    # It may come with a performance cost, as noted here:
    # https://github.com/geopandas/geopandas/issues/1820
    # https://github.com/geopandas/geopandas/issues/2171
    # This issue may disappear when shapely 2.0 is released (then pygeos
    # is merged with shapely).
    mergedlines = unary_union(list(all_lines))
    # mergedlines = unary_union(all_lines)  # TODO Try this with shapely 2.0

    # empty geopandas dataframe for house connections
    conn_lines = gpd.GeoDataFrame(geometry=[], crs=lines.crs)

    # iterate over all houses
    for index, row in points.iterrows():

        house_geo = row['geometry']

        # new nearest point method  ############ #########
        n_p = nearest_points(mergedlines, house_geo)[0]

        # get index of line which is closest to the house
        line_index = line_of_point(n_p, lines)

        # get geometry of supply line
        supply_line = lines.loc[line_index, 'geometry']

        # get end points of line
        supply_line_p0 = Point(list(supply_line.coords)[0])
        supply_line_p1 = Point(list(supply_line.coords)[1])
        supply_line_points = [supply_line_p0, supply_line_p1]
        supply_line_mulitpoints = MultiPoint(supply_line_points)

        if n_p in supply_line_points:
            # case that nearest point is a line ending

            logger.debug(
                'Connect buildings... id {}: '
                'Connected to supply line ending (nearest point)'.format(index)
            )

            con_line = LineString([n_p, house_geo])

            conn_lines = pd.concat(
                [gpd.GeoDataFrame(conn_lines, crs=lines.crs),
                 gpd.GeoDataFrame(geometry=[con_line], crs=lines.crs)],
                ignore_index=True)

        else:

            dist_to_endings = [x.distance(n_p) for x in supply_line_points]

            if min(dist_to_endings) >= tol_distance:
                # line is split, no line ending is close to the nearest point
                # this also means the original supply line needs to be deleted

                logger.debug(
                    'Connect buildings... id {}: Supply line split'.format(index))

                con_line = LineString([n_p, house_geo])

                conn_lines = pd.concat(
                    [gpd.GeoDataFrame(conn_lines, crs=lines.crs),
                     gpd.GeoDataFrame(geometry=[con_line], crs=lines.crs)],
                    ignore_index=True)

                lines.drop([line_index], inplace=True)

                lines = pd.concat(
                    [gpd.GeoDataFrame(lines, crs=lines.crs),
                     gpd.GeoDataFrame(geometry=[
                         LineString([supply_line_p0, n_p]),
                         LineString([n_p, supply_line_p1])], crs=lines.crs)],
                    ignore_index=True)

            else:
                # case that one or both line endings are closer than tolerance
                # thus, the next line ending is chosen
                logger.debug(
                    'Connect buildings... id {}: Connected to Supply line '
                    'ending due to tolerance'.format(index))

                conn_point = nearest_points(supply_line_mulitpoints, n_p)[0]

                con_line = LineString([conn_point, house_geo])

                conn_lines = pd.concat(
                    [gpd.GeoDataFrame(conn_lines, crs=lines.crs),
                     gpd.GeoDataFrame(geometry=[con_line], crs=lines.crs)],
                    ignore_index=True)

    logger.info('Connection of buildings completed.')

    return conn_lines, lines


def check_geometry_type(gdf, types):
    """
    Checks, if a geodataframe has only the given geometry types in its GeoSeries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        DataFrame to be checked.
    types : list
        List of types allowed for GeoDataFrame.

    """
    actual_types = set(gdf['geometry'].type)

    for type in actual_types:
        if type not in types:
            raise TypeError(
                "Your input geometry has the wrong type. "
                "Expected: {}. Got: {}".format(types, type)
            )


def create_points_from_polygons(gdf, method='midpoint'):
    """
    Converts the geometry of a polygon layer to a point layer.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
    method : str
        Method to create a point from a polygon.

    Returns
    -------
    geopandas.GeoDataFrame : GeoDataFrame with a point geometry.

    """

    if gdf['geometry'].values[0].geom_type == 'Point':
        return gdf

    if method == 'midpoint' or method == 'boundary':
        # (method 'boundary' is performed later and needs the centroid)
        gdf['geometry'] = gdf['geometry'].centroid
        return gdf

    raise ValueError(
        "No other methods than 'midpoint' and 'boundary' implemented!"
    )


def run_point_method_boundary(consumers_poly, consumers, lines_consumers):
    """Run 'boundary' method for finding the building connection point.

    This is meant to be called once with the consumers as input and
    once with the producers (producers_poly, producers, lines_producers).

    The 'midpoint' method (using the centroid) must already have been run,
    generating the default connection lines from street to centroid.

    If there is only one intersection between that line and the boundary
    of the building, this intersection point is used as the connection
    point instead (and the connection line is shortened accordingly).

    However, complex building shapes can produce multiple intersections. In
    this case, the intersection with the 'convex hull' of the building is used
    instead. This may result in connection points that do not touch an
    actual building wall, but it should still be an improvement compared to
    the 'midpoint' method.

    In case of no intersections with the building boundary (possible for e.g.
    U-shaped buildings), the original centroid is used. This is also
    necessary when the the street line already touches the building, to
    prevent deleting the connection line.

    Parameters
    ----------
    consumers_poly : geopandas.GeoDataFrame
        Polygons of the consumer buildings. Point geometries are also allowed,
        but they are not changed.
    consumers : geopandas.GeoDataFrame
        Points of the consumer buildings (as returned by 'midpoint' method).
    lines_consumers : geopandas.GeoDataFrame
        Connection lines from street to each consumer point.

    Returns
    -------
    consumers : geopandas.GeoDataFrame
        Updated points of the consumer buildings.
    lines_consumers : geopandas.GeoDataFrame
        Updated connection lines from street to each consumer point.

    """
    logger.info('Run "boundary" method for finding the building connections')
    # lines_consumers may represent multiple lines per consumer
    # Duplicate geometries in consumers_poly have to be created accordingly
    consumers_poly['id_full'] = consumers['id_full']
    consumers_poly = pd.merge(
        left=consumers_poly,
        right=lines_consumers.drop(columns=[lines_consumers.geometry.name]),
        how='right', on='id_full')

    # When using the original consumer points as a fallback later, we
    # require it to have the same index as lines_consumers. Therefore
    # we create the 'duplicate' consumers object
    consumers_d = pd.merge(
        left=consumers,
        right=lines_consumers.drop(columns=[lines_consumers.geometry.name]),
        how='right', on='id_full')

    # Cut the part off of each "line_consumer" that is within the building
    # polygon. As a result, the heating grid will only reach to the wall of
    # the building.
    lines_consumers_n = gpd.GeoDataFrame(
        geometry=lines_consumers.difference(consumers_poly, align=False))
    # This produces a MultiLineString for complex building polygons, where
    # the boundary and the simple lines from centroid to street intersect
    # multiple times. This would not be a valid connection line. In those
    # cases the 'convex hull' of the building is used instead.
    lines_consumers_n.loc[lines_consumers_n.type == "MultiLineString"] = \
        gpd.GeoDataFrame(geometry=lines_consumers.difference(
            consumers_poly.convex_hull, align=False))
    # Only keep the new consumer lines if they have a useful minimum length.
    # There was an edgecase where a street 'almost' touched a building,
    # and the cut consumer line had a length of 1e-9 m
    lines_consumers_n[lines_consumers_n.length < 1e-3] = LineString()

    # Now the "consumers" (point objects for each building) need to be
    # updated to touch the end of the consumer_lines
    consumers_n = gpd.GeoDataFrame(geometry=lines_consumers.intersection(
        consumers_poly.boundary, align=False))
    consumers_n.loc[consumers_n.type == "MultiPoint"] = \
        gpd.GeoDataFrame(geometry=lines_consumers.intersection(
            consumers_poly.convex_hull.boundary, align=False))

    # Sometimes the centroid does not lie within a building and there may be
    # no intersetions, i.e. the new point is an 'empty' geometry. This can
    # happen if buildings are multipolygons, which is not forbidden.
    # Sometimes the new lines are empty (e.g. because a street and a building
    # object cross each other).
    # In these cases the original geometry is used for points and lines.
    mask1 = (consumers_n.is_empty | lines_consumers_n.is_empty)

    # Another special case has to be covered. If the original street lines
    # already touch the building wall, no additional connection line would be
    # necessary. However, in dhnx each building needs one connection line.
    # Thus the geometry from the 'midpoint' method is used here, too.
    # Find the problematic cases by testing if the new connection point
    # equals the starting point of the connection line.
    mask2 = consumers_n.geom_equals(
        lines_consumers.geometry.apply(lambda line: line.boundary.geoms[0]))

    # If for whatever reason the street-side "start" of the new connection
    # line is not the same point as the original connection line start, use
    # the original line. This may happen for complex geometries, where the
    # street line lies within the building geometry
    lines_consumers_n_start = lines_consumers_n.copy()
    lines_consumers_n_start.geometry = (
        lines_consumers_n_start[~mask1 & ~mask2].boundary.apply(
            lambda g: g.geoms[0]))
    mask3 = lines_consumers_n_start.geom_equals(
        lines_consumers.geometry.apply(lambda line: line.boundary.geoms[0]))

    # Now apply all the filters above to reset the geometries
    mask = mask1 | mask2 | ~mask3
    consumers_n.loc[mask] = consumers_d.loc[mask].geometry
    lines_consumers_n.loc[mask] = lines_consumers.loc[mask].geometry

    # Now update all the original variables with the new data
    lines_consumers_n['id_full'] = lines_consumers['id_full']
    consumers_n['id_full'] = lines_consumers_n['id_full']

    # If multiple building connection lines existed before, we now also
    # have created multiple building points for each building.
    # We need to group those points into one MultiPoint per initial unique
    # building, to keep the original index structure intact.
    consumers_n = (pd.DataFrame(consumers_n)  # convert gdf to df
                   .groupby('id_full', sort=False, as_index=False)
                   .agg(lambda x: MultiPoint(x.values))  # returns df
                   .set_geometry(consumers_n.geometry.name,  # convert to gdf
                                 crs=consumers_n.crs)
                   )

    # For each new consumer point(s), test if they actually touch
    # the new conumser line(s) that have the same 'id_full' assigned
    for id_full, points in zip(consumers_n['id_full'], consumers_n.geometry):
        if not points.touches(lines_consumers_n[
                lines_consumers_n['id_full'] == id_full].geometry).all():
            raise ValueError(f"Points from {id_full} have no matching lines")

    consumers.geometry = consumers_n.geometry
    lines_consumers = lines_consumers_n
    return consumers, lines_consumers


def check_duplicate_geometries(gdf):
    """Test the input GeoDataFrame for duplicate geometries and plot them."""
    if gdf.duplicated(subset="geometry").any():
        idx = gdf.duplicated(subset="geometry")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(dpi=400)
            gdf.loc[~idx].plot(ax=ax, color='green')
            gdf.loc[idx].plot(ax=ax, color='red')
            plt.title("Red are duplicate geometries. Please fix!")
            plt.show()
        except ImportError:
            logger.info("Install matplotlib to show a plot of the duplicate "
                        "geometries.")
        raise ValueError("GeoDataFrame has {} duplicate geometries"
                         .format(len(gdf.loc[idx])))


def process_geometry(lines, consumers, producers,
                     method='midpoint', projected_crs=4647,
                     tol_distance=2, reset_index=True,
                     welding=True):
    """
    This function connects the consumers and producers to the line network, and prepares the
    attributes of the geopandas.GeoDataFrames for importing as dhnx.ThermalNetwork.

    The ids of the lines are overwritten.

    Parameters
    ----------
    lines : geopandas.GeoDataFrame
        Potential routes for the DHS. Expected geometry Linestrings or MultilineStrings.
        The graph of this line network should be connected.
    consumers : geopandas.GeoDataFrame
        Location of demand/consumers. Expected geometry: Polygons or Points.
    producers : geopandas.GeoDataFrame
        Location of supply sites. Expected geometry: Polygons or Points.
    method : str
        Method for creating the point if polygons are given for the consumers
        and producers. Method 'midpoint' uses the centroid of each building
        polygon. Method 'boundary' moves the point to the boundary (wall) of
        the building, along the line constructed from centroid to the street.
    multi_connections : bool
        Setting if a building should be connected to multiple streets.
    projected_crs : EPSG integer code
        EPSG Coordinate reference system number (eg 4647),
        which is used for the geometry operations.
        A projected crs must be used!
    tol_distance : float
        Tolerance distance at connection the points to the line network
        for choosing the end of the line instead of the lot.
    reset_index : bool
        If True, reset the index and ignore the existing index. If False,
        use the existing index for consumer and producer identificators.
        Default: True
    welding : bool, optional
        Weld continuous line segments together and cut loose ends. This
        can improve the performance of the optimization, as it decreases
        the total number of line elements. Default is True.

    Returns
    -------
    dict : Dictionary with 4 geopandas.GeoDataFrames: The keys of the Dict are
           equal to the components of the dhnx.ThermalNetwork: 'forks', 'consumers',
           'producers', 'pipes'.

    """
    if not reset_index:
        raise ValueError("Keeping the orginal index is currently not "
                         "supported. Use 'reset_index=True'.")

    # Copies of the original polygons are needed for method 'boundary'
    consumers_poly = go.check_crs(consumers, crs=projected_crs).copy()
    producers_poly = go.check_crs(producers, crs=projected_crs).copy()

    # check whether the expected geometry is used for geo dataframes
    check_geometry_type(lines, types=['LineString', 'MultiLineString'])
    for gdf in [producers, consumers, producers_poly, consumers_poly]:
        check_geometry_type(gdf, types=['Polygon', 'Point', 'MultiPolygon'])
        check_duplicate_geometries(gdf)

    # split multilinestrings to single lines with only 1 starting and 1 ending point
    lines = go.split_multilinestr_to_linestr(lines)

    # check and convert crs if it is not already the `projected_crs`
    lines = go.check_crs(lines, crs=projected_crs)

    for layer in [producers, consumers]:
        layer = go.check_crs(layer, crs=projected_crs)
        layer = create_points_from_polygons(layer, method=method)
        layer['lat'] = layer['geometry'].apply(lambda x: x.y)
        layer['lon'] = layer['geometry'].apply(lambda x: x.x)

    for layer in [producers, consumers, producers_poly, consumers_poly]:
        if reset_index:
            layer.reset_index(inplace=True, drop=True)
            layer.index.name = 'id'
            layer.drop(columns=['id'], inplace=True, errors='ignore')
        else:
            if layer.index.has_duplicates:
                raise ValueError("The index of input data has duplicate "
                                 "values, which is not allowed")

    producers['id_full'] = 'producers-' + producers.index.astype('str')
    producers['type'] = 'G'
    consumers['id_full'] = 'consumers-' + consumers.index.astype('str')
    consumers['type'] = 'H'

    # Add lines to consumers and producers
    lines_consumers, lines = create_object_connections(
        consumers, lines, tol_distance=tol_distance)
    lines_producers, lines = create_object_connections(
        producers, lines, tol_distance=tol_distance)
    if not reset_index:
        # Connection lines are ordered the same as points. Match their indexes
        lines_consumers.index = consumers.index
        lines_producers.index = producers.index

    if method == 'boundary':
        # Can only be performed after 'midpoint' method
        consumers, lines_consumers = run_point_method_boundary(
            consumers_poly, consumers, lines_consumers)
        producers, lines_producers = run_point_method_boundary(
            producers_poly, producers, lines_producers)

    if welding:
        # Weld continuous line segments together and cut loose ends
        lines = go.weld_segments(
            lines, lines_producers, lines_consumers,
            # debug_plotting=True,
        )

    # Keep only the shortest of all lines connecting the same two points
    lines = go.drop_parallel_lines(lines)

    # add additional line identifier
    lines_producers['type'] = 'GL'  # GL for generation line
    lines['type'] = 'DL'  # DL for distribution line
    lines_consumers['type'] = 'HL'  # HL for house line

    # generate forks point layer
    forks = go.create_forks(lines)

    # concat lines
    lines_all = pd.concat([lines, lines_consumers, lines_producers], sort=False)
    lines_all.reset_index(inplace=True, drop=True)
    if reset_index:
        lines_all.index.name = 'id'
        lines_all.drop(columns=['id'], inplace=True, errors='ignore')

    # concat point layer
    points_all = pd.concat([
        consumers[['id_full', 'geometry']],
        producers[['id_full', 'geometry']],
        forks[['id_full', 'geometry']]],
        sort=False
    )

    # add from_node, to_node to lines layer
    lines_all = go.insert_node_ids(lines_all, points_all)

    lines_all['length'] = lines_all.length
    logger.info(
        "Total line length is {:.0f} m".format(lines_all['length'].sum()))

    # Convert all MultiLineStrings to LineStrings
    check_geometry_type(lines_all, types=['LineString'])

    # ## check for near points
    go.check_double_points(points_all, id_column='id_full')

    return {
        'forks': forks,
        'consumers': consumers,
        'producers': producers,
        'pipes': lines_all,
    }
