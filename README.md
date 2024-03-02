# xarray backend for Selafin formats

## Dev guide

to have the backend working in xarray, follow these steps

```
pip install xarray-selafin
```

## Read selafin

```python
import xarray as xr
ds = xr.open_dataset("input_file.slf", engine="selafin")
```

## Write selafin

```python
ds.selafin.write("output_file.slf")
```

## DataSet content

### Dimensions
* time
* node
* plan (only in 3D)
 
### Coordinates

| Coordinate | Description            |
|------------|------------------------|
| x          | East mesh coordinates  |
| y          | North mesh coordinates |
| time       | Datetime serie         |

### Attributes

All attributes are optional except `ikle2`:

| Attribute  | Description                                                              | Default value            | 
|------------|--------------------------------------------------------------------------|--------------------------|
| title      | Serafin title                                                            | "" (empty string)        |
| float_size | Float size                                                               | 4 (single precision)     |
| endian     | File endianness                                                          | ">"                      |
| params     | Table of integer parameters                                              | (can be rebuilt)         |
| ikle2      | Connectivity table in 2D (1-indexed)                                     | -                        |
| ikle3      | Connectivity table in 3D (1-indexed, only in 3D, optional)               | (can be rebuilt from 2D) |
| variables  | Dictionary with variable names and units (key is variable abbreviation)  | -                        |
| date_start | Starting date with integers                                              | (from first time serie)  |
