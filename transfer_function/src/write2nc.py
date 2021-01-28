import numpy as np
from netCDF4 import Dataset

def write2nc(filename, x, y, value, name):
    print("Writing to a .nc file")

    dataset_new = Dataset(filename, 'w', format='NETCDF4_CLASSIC')

    dataset_new.createDimension('lat', len(y))
    dataset_new.createDimension('lon', len(x))

    wave_height = dataset_new.createVariable(str(name), np.float64, ('lat','lon'))
    lat = dataset_new.createVariable('lat', np.float64,'lat')
    lon = dataset_new.createVariable('lon', np.float64,'lon')

    lat.units = 'm'
    lon.units = 'm'
    wave_height.units = 'm'

    lat[:] = y
    lon[:] = x
    wave_height[:] = value

    # Close the netcdf file
    dataset_new.close()

    return
