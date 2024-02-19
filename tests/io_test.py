import pytest
import xarray as xr


TIDAL_FLATS = pytest.mark.parametrize(
    "slf_in",
    [
        pytest.param('tests/data/r3d_tidal_flats.slf', id="3D"),
        pytest.param('tests/data/r2d_tidal_flats.slf', id="2D"),
    ]
)

@TIDAL_FLATS
def test_open_dataset(slf_in):
    ds = xr.open_dataset(slf_in, engine='selafin')
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords


@TIDAL_FLATS
def test_to_netcdf(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, engine='selafin')
    nc_out = tmp_path / "test.nc"
    ds_slf.to_netcdf(nc_out)
    ds_nc = xr.open_dataset(nc_out)
    assert ds_nc.equals(ds_slf)
