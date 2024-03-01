import pytest
import xarray as xr


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

    # Dimensions
    assert ds.sizes['time'] == 17
    assert ds.sizes['node'] == 648
    assert ds.sizes['elem2'] == 1030
    if "r3d" in slf_in:
        assert ds.sizes['plan'] == 21
        assert ds.sizes['elem3'] == 20600
        assert ds.sizes['ndp3'] == 6
    else:
        assert ds.sizes['ndp2'] == 3

    # Coordinates
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords
    assert "ikle2" in ds.coords
    if "r3d_" in slf_in:
        assert "ikle3" in ds.coords
    else:
        assert "ikle3" not in ds.coords

    # Attributes
    assert ds.attrs["endian"] == ">"
    assert ds.attrs["date_start"] == (1900, 1, 1, 0, 0, 0)


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

    # Compare binary files
    with open(slf_in, 'rb') as in_slf1, open(slf_out, 'rb') as in_slf2:
        assert in_slf1.read() == in_slf2.read()


@TIDAL_FLATS
def test_to_selafin_eager_mode(tmp_path, slf_in):
    ds_slf = xr.open_dataset(slf_in, lazy_loading=False, engine="selafin")
    slf_out = tmp_path / "test.slf"
    ds_slf.selafin.write(slf_out)
    ds_slf2 = xr.open_dataset(slf_out, engine="selafin")
    assert ds_slf2.equals(ds_slf)

    # Compare binary files
    with open(slf_in, 'rb') as in_slf1, open(slf_out, 'rb') as in_slf2:
        assert in_slf1.read() == in_slf2.read()


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
    if "r3d" in slf_in:
        # multiple slices
        ds_slf = xr.open_dataset(slf_in, engine="selafin")
        nc_out = tmp_path / "test3.nc"
        ds_slice = ds_slf.isel(time=slice(0, 10), plan=0)
        ds_slice.to_netcdf(nc_out)
        ds_nc = xr.open_dataset(nc_out)
        assert ds_nc.equals(ds_slice)
        # multiple range slices
        ds_slf = xr.open_dataset(slf_in, engine="selafin")
        nc_out = tmp_path / "test4.nc"
        ds_slice = ds_slf.isel(time=slice(0, 10), plan=slice(5, 10))
        ds_slice.to_netcdf(nc_out)
        ds_nc = xr.open_dataset(nc_out)
        assert ds_nc.equals(ds_slice)
