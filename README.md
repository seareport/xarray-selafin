# xarray backend for Selafin formats

## Dev guide

to have the backend working in xarray, follow these steps

```
git clone this repository
poetry install 
```

## Read selafin

```python
import xarray as xr
ds = xr.open_dataset(file_3d, engine='selafin')
```
## Write selafin

```python
ds.selafin.write('test.nc')
```
