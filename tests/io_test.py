import pytest
import xarray as xr
import numpy as np

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

TIDAL_FLATS = pytest.mark.parametrize(
    "slf_in",
    [
        pytest.param("tests/data/r3d_tidal_flats.slf", id="3D"),
        pytest.param("tests/data/r2d_tidal_flats.slf", id="2D"),
    ],
)


@TIDAL_FLATS
def test_open_dataset(slf_in):
    ds = xr.open_dataset(slf_in, engine="selafin")
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords


@TIDAL_FLATS
def test_to_netcdf(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test.nc"
    ds_slf.to_netcdf(nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slf)


@TIDAL_FLATS
def test_to_selafin(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    slf_out = tmp_path / "test.slf"
    ds_slf.selafin.write(slf_out)
    ds_slf2 = xr.open_dataset(slf_out, engine="selafin")
    assert ds_slf2.equals(ds_slf)


@TIDAL_FLATS
def test_slice(tmp_path, slf_in):
    # simple selection
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test1.nc"
    ds_slice = ds_slf.isel(time=0)
    ds_slice.to_netcdf(nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slice)
    # simple range
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test2.nc"
    ds_slice = ds_slf.isel(time=slice(0, 10))
    ds_slice.to_netcdf(nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slice)
    # multiple slices
    ds_slf = xr.open_dataset(slf_in, engine="selafin")
    nc_out = tmp_path / "test3.nc"
    ds_slice = ds_slf.isel(time=slice(0, 10), plan=0)
    ds_slice.to_netcdf(nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slice)
    # # multiple range slices  # FIXME: not working yet
    # ds_slf = xr.open_dataset(slf_in, engine="selafin")
    # nc_out = tmp_path / "test3.nc"
    # ds_slice = ds_slf.isel(time=slice(0, 10), plan=slice(5, 10))
    # ds_slice.to_netcdf(nc_out)
    # ds_nc = xr.open_dataset(nc_out)
    # assert ds_nc.equals(ds_slice)
