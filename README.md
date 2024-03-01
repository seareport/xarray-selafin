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
* elem2
* ndp2
* elem3 (only in 3D)
* ndp3 (only in 3D)
 
### Coordinates

| Coordinate | Description                                     |
|------------|-------------------------------------------------|
| x          | East coordinates                                |
| y          | North coordinates                               |
| time       | Datetime serie                                  |
| ikle2      | Connectivity table in 2D                        |
| ikle3      | Connectivity table in 3D (only in 3D, optional) |

### Attributes

All attributes are optional:

| Attribute  | Description                                    | Default value             | 
|------------|------------------------------------------------|---------------------------|
| title      | Serafin title                                  | (empty)                   |
| float_size | Float size                                     | 4 (single precision)      |
| endian     | File endianness                                | ">"                       |
| params     | Table of integer parameters                    | (can be rebuilt)          |
| var_IDs    | List of variable identifiers                   |                           |
| varnames   | List of variable names (same order of var_IDs) | (use variable identifier) |
| varunits   | List of variable units (same order of var_IDs) | ?                         |
| date_start | Starting date with integers                    | (from first time serie)   |
