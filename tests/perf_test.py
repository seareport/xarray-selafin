import time
from xarray_selafin_backend.selafin import Selafin
from xarray_selafin_backend.telemac_file import TelemacFile
import pytest

PERF = pytest.mark.parametrize(
    "f",
    [
        pytest.param("tests/data/r3d_tidal_flats.slf", id="3D"),
        pytest.param("tests/data/r2d_tidal_flats.slf", id="2D"),
    ],
)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end - start} seconds")
        return result
    return wrapper
   
# selafin
@timer
@PERF 
def selafin(f):
    slf = Selafin(f)
    slf.get_series([10])

# telemacFile
@timer
@PERF 
def telemac(f):
    tel = TelemacFile(f)
    tel.get_data_value(tel.varnames[0], 10)
