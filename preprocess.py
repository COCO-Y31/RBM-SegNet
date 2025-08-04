import os
from numpy import random
import warnings
import rvt.vis
import rvt.default
warnings.filterwarnings("ignore")


from scipy import signal
import richdem
from osgeo import gdal


def readtif(DEM_filepath, is_resample=False, scale=1):
    dataset = gdal.Open(DEM_filepath)
    data = dataset.ReadAsArray()
    data = np.around(data, 3)             

    geotrans = list(dataset.GetGeoTransform())
    proj = dataset.GetProjection()
    band_count = dataset.RasterCount
    data_type = dataset.GetRasterBand(1).DataType

    del dataset
    return data, proj, geotrans, data_type


def fill_depression(dem_map):
    """
    Args:
         dem     (rdarray): An elevation model
         epsilon (float):   If True, an epsilon gradient is imposed to all flat regions.
                            This ensures that there is always a local gradient.
         in_place (bool):   If True, the DEM is modified in place and there is
                            no return; otherwise, a new, altered DEM is returned.                                     
         topology (string): A topology indicator
    """
    dem_rd = richdem.rdarray(dem_map, no_data=-9999)
    dem_without_depression = np.array(richdem.FillDepressions(dem_rd))

    return dem_without_depression


def get_constant2onezero(dem_img):
    """
    dem_img: numpy_array, (H, W) into model fine_size
    :return: (H, W)
    """
    max_val = dem_img.max()
    min_val = dem_img.min()
    dem_img = (dem_img - min_val)/(max_val - min_val)

    return dem_img


def get_slope(dem_map, padding):
    slope_i, slope_j = np.gradient(dem_map)
    slope = np.sqrt(slope_i**2 + slope_j**2)
    slope[np.isnan(slope)] = 0
    return slope[padding:padding*(-1), padding:padding*(-1)]

import richdem as rd
def get_aspect(dem_map, padding):
    dem_map = rd.LoadGDAL(dem_map, no_data=-9999)
    aspect = rd.TerrainAttribute(dem_map, attrib='aspect')
    return aspect[padding:padding * (-1), padding:padding * (-1)]

def get_hillsahde(dem_path, padding):
    dict_dem = rvt.default.get_raster_arr(dem_path)
    dem_arr = dict_dem["array"]  # numpy array of DEM
    dem_resolution = dict_dem["resolution"]
    dem_res_x = dem_resolution[0]  # resolution in X direction
    dem_res_y = dem_resolution[1]  # resolution in Y direction
    sun_azimuth = 315  # Solar azimuth angle (clockwise from North) in degrees
    sun_elevation = 45  # Solar vertical angle (above the horizon) in degrees
    hillshade_arr = rvt.vis.hillshade(dem=dem_arr, resolution_x=dem_res_x, resolution_y=dem_res_y,
                                      sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, ve_factor=1)

    return hillshade_arr[padding:padding * (-1), padding:padding * (-1)]
