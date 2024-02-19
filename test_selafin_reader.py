import xarray as xr
import matplotlib.pyplot as plt
import time
from data_manip.formats.selafin import Selafin
from data_manip.extraction.telemac_file import TelemacFile


def main():
    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Time taken by {func.__name__}: {end - start} seconds")
            return result

        return wrapper

    @timer
    def ppUtils(file_path, ax):
        # read test
        ds = xr.open_dataset(file_path, engine="selafin")
        ax.triplot(ds.x, ds.y, ds.ikle[:, :3])

        # to netcdf -- NOT WORKING
        ds1 = ds.isel(time=0)
        ds1.to_netcdf("test1.nc")
        ds.to_netcdf("test.nc")

    @timer
    def selafin(file_path, ax):
        slf = Selafin(file_path)
        ax.triplot(slf.meshx, slf.meshy, slf.ikle2)

    @timer
    def telemac(file_path):
        tel = TelemacFile(file_path)

    file_3d = "tests/data/r3d_tidal_flats.slf"
    # file_3d = '/home/tomsail/Documents/work/python/pyPoseidon/Tutorial/test/hindcast/201304/input_wind_tmp.slf'

    fig, ax = plt.subplots(figsize=(26, 13))
    ppUtils(file_3d, ax)
    ax.set_aspect("equal")
    plt.show()
    selafin(file_3d, ax)
    telemac(file_3d)


main()


# # write -- NOT WORKING
# ds.selafin.write('test.slf')
