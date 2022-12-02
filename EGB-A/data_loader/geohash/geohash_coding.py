#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:35:08 2018

@author: tang
"""
#import gdal
#from osgeo import osr


def encode(latitude, longitude, precision=20):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the pointed precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    geohash = []

    is_longitude = True

    while len(geohash) < precision:
        if is_longitude:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                geohash.append(1)
                lon_interval = (mid, lon_interval[1])
            else:
                geohash.append(0)
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                geohash.append(1)
                lat_interval = (mid, lat_interval[1])
            else:
                geohash.append(0)
                lat_interval = (lat_interval[0], mid)

        is_longitude = not is_longitude

    return geohash


# def get_top_left_latlng(img_path):
#     """
#     Given a GDAL dataset, computes lat/lng of its top left corner.
#     """
#     dataset = gdal.Open(img_path)
#     wgs84_spatial_reference = osr.SpatialReference()
#     wgs84_spatial_reference.ImportFromEPSG(4326)

#     dataset_spatial_reference = osr.SpatialReference()
#     dataset_spatial_reference.ImportFromWkt(dataset.GetProjection())

#     dataset_to_wgs84 = osr.CoordinateTransformation(dataset_spatial_reference,
#                                                     wgs84_spatial_reference)

#     geo_transform = dataset.GetGeoTransform()

#     x_geo = geo_transform[0]
#     y_geo = geo_transform[3]
#     lng, lat, _ = dataset_to_wgs84.TransformPoint(x_geo, y_geo)

#     return lat, lng
