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
ds = xr.open_dataset("input_file.slf", engine='selafin')
```
## Write selafin

```python
ds.selafin.write('output_file.slf')
```
