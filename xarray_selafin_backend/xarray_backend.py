from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core import indexing
import numpy as np
import xarray as xr
import os
import logging

from xarray_selafin_backend import Serafin

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def read_serafin(f):
    resin = Serafin.Read(f, "en")
    resin.__enter__()
    resin.read_header()
    resin.get_time()
    return resin


def write_serafin(fout, ds):
    slf_header = Serafin.SerafinHeader("Converted from Xarray")
    slf_header.date = ds.attrs["date"]
    slf_header.float_type = "d"
    slf_header.float_size = 8
    slf_header.np_float_type = np.float64
    slf_header.endian = ">"

    slf_header.nb_var = len(ds.varnames)
    if "plan" in ds.dims:
        slf_header.nb_planes = len(ds.plan)
        if slf_header.nb_planes == 1:
            slf_header.nb_planes = 0
    else:
        slf_header.nb_planes = 0

    if slf_header.nb_planes > 1:
        slf_header.is_2d = False
    else:
        slf_header.is_2d = True
    slf_header.nb_frames = len(ds.time)

    slf_header.var_IDs = ds.var_IDs
    slf_header.var_names = [s.ljust(16).encode("utf-8") for s in ds.varnames]
    slf_header.var_units = ds.varunits

    slf_header.params = list(ds.attrs["iparam"])
    slf_header.nb_elements = ds.attrs["nelem3"]
    slf_header.nb_nodes = ds.attrs["npoin3"]
    slf_header.nb_nodes_per_elem = ds.attrs["ndp3"]
    slf_header.nb_nodes_2d = ds.attrs["npoin2"]

    slf_header.x = ds.attrs["meshx"]
    slf_header.y = ds.attrs["meshy"]
    slf_header.mesh_origin = (0.0, 0.0)
    slf_header.x_stored = slf_header.x - slf_header.mesh_origin[0]
    slf_header.y_stored = slf_header.y - slf_header.mesh_origin[1]
    slf_header.ikle = ds["ikle3"].data.ravel()
    slf_header.ikle_2d = ds["ikle2"].data
    slf_header.ipobo = ds.attrs["ipob3"]
    vars = [bs.decode("utf-8") for bs in slf_header.var_names]

    resout = Serafin.Write(fout, "en", overwrite=True)
    resout.__enter__()
    resout.write_header(slf_header)
    shape = (
        (slf_header.nb_var, slf_header.nb_nodes_2d, slf_header.nb_planes)
        if slf_header.nb_planes > 1
        else (slf_header.nb_var, slf_header.nb_nodes_2d)
    )
    for it, t_ in enumerate(ds.time):
        temp = np.zeros(shape, dtype=slf_header.np_float_type)
        for iv, var in enumerate(vars):
            temp[iv, :] = ds.isel(time=it)[var.strip()].squeeze()
        resout.write_entire_frame(
            slf_header,
            t_,
            np.reshape(np.ravel(temp), (slf_header.nb_var, slf_header.nb_nodes)),
        )


class SelafinLazyArray(BackendArray):
    def __init__(self, slf_reader, var, dtype, shape):
        self.slf_reader = slf_reader
        self.var = var
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key):
        #     return indexing.explicit_indexing_adapter(
        #         key,
        #         self.shape,
        #         indexing.IndexingSupport.BASIC,
        #         self._raw_indexing_method,
        #     )

        # def _raw_indexing_method(self, key):
        logging.debug("Raw indexing method called")

        if isinstance(key, tuple) and len(key) == 3:
            time_key, node_key, plan_key = key
        elif isinstance(key, tuple) and len(key) == 2:
            time_key, node_key = key
            # Default plan_key to select all if not provided
            plan_key = slice(None)
        else:
            # If a single key is provided, it's assumed for the time dimension
            time_key = key
            # Default node_key and plan_key to select all
            node_key = slice(None)
            plan_key = slice(None)

        # Convert time_key and node_key to ranges to handle steps and to list indices for SELAFIN reader
        if isinstance(time_key, slice):
            time_indices = range(*time_key.indices(self.shape[0]))
        elif isinstance(time_key, int):
            time_indices = [time_key]
        else:
            raise ValueError("time_key must be an integer or slice")

        if isinstance(node_key, slice):
            node_indices = range(*node_key.indices(self.shape[1]))
        elif isinstance(node_key, int):
            node_indices = [node_key]
        else:
            raise ValueError("time_key must be an integer or slice")

        if isinstance(plan_key, slice):
            plan_indices = range(*plan_key.indices(self.shape[2]))
        elif isinstance(plan_key, int):
            plan_indices = [plan_key]
        else:
            raise ValueError("time_key must be an integer or slice")

        # Initialize data array to hold the result
        data_shape = (len(time_indices), len(node_indices), len(plan_indices))
        data = np.ones(data_shape, dtype=self.dtype)
        vars = [bs.decode("utf-8").strip() for bs in self.slf_reader.header.var_names]
        var_idx = vars.index(self.var)
        # Iterate over the time indices to read the required time steps
        for it, t in enumerate(time_indices):
            temp = self.slf_reader.read_vars_in_frame(t)[var_idx]
            temp = np.reshape(temp, self.shape[1:])

            if isinstance(plan_key, int):
                # Handling case where node_key is an integer (selecting a single node across time)
                if isinstance(node_key, int):
                    data[it] = temp[node_key, plan_key]
                else:
                    # For each time step, extract the required nodes based on node_key
                    for j, n_idx in enumerate(node_indices):
                        data[it, j] = temp[n_idx, plan_key]
            else:
                for p, p_idx in enumerate(plan_indices):
                    if isinstance(node_key, int):
                        data[it, :, p] = temp[node_key, p_idx]
                    else:
                        # For each time step, extract the required nodes based on node_key
                        for j, n_idx in enumerate(node_indices):
                            data[it, j, p] = temp[n_idx, p_idx]

        return data.squeeze()


class SelafinBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
    ):
        # Initialize SELAFIN reader
        slf = read_serafin(filename_or_obj)

        # Prepare dimensions, coordinates, and data variables
        times = slf.time
        nelem2 = len(slf.header.ikle_2d)
        nelem3 = slf.header.nb_elements
        npoin2 = slf.header.nb_nodes_2d
        npoin3 = slf.header.nb_nodes
        ndp2 = 3
        ndp3 = slf.header.nb_nodes_per_elem
        nplan = max(1, slf.header.nb_planes)
        ikle2 = slf.header.ikle_2d
        ikle3 = np.reshape(slf.header.ikle, (nelem3, ndp3))
        ipob2 = slf.header.ipobo
        ipob3 = slf.header.ipobo
        x = slf.header.x
        y = slf.header.y
        vars = [bs.decode("utf-8").strip() for bs in slf.header.var_names]

        # Create data variables using Dask arrays for the variables
        data_vars = {}
        dtype = np.float64
        shape = (len(times), npoin2, nplan)

        if DASK_AVAILABLE:
            for name in vars:
                lazy_array = SelafinLazyArray(slf, name, dtype, shape)
                dask_array = da.from_array(lazy_array, chunks=(1, shape[1], shape[2]))
                data_vars[name.strip()] = (["time", "node", "plan"], dask_array)

        else:
            for name in vars:
                var_idx = vars.index(name)
                data = np.zeros((len(times), npoin2, nplan), dtype=dtype)
                for it, t in enumerate(times):
                    variable_data = slf.read_vars_in_frame(it)[var_idx]
                    data[it, :] = np.reshape(variable_data, (npoin2, nplan))

                data_vars[name.strip()] = (["time", "node", "plan"], data)

        coords = {
            "x": ("node", x[:npoin2]),
            "y": ("node", y[:npoin2]),
            "time": times,
            # Adding IKLE as a coordinate or data variable for mesh connectivity
            "ikle2": (("nelem2", "ndp2"), ikle2),
            "ikle3": (("nelem3", "ndp3"), ikle3),
            # Consider how to include IPOBO if it's essential for your analysis
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        ds.attrs["meshx"] = x
        ds.attrs["meshy"] = y
        ds.attrs["nelem2"] = nelem2
        ds.attrs["nelem3"] = nelem3
        ds.attrs["npoin2"] = npoin2
        ds.attrs["npoin3"] = npoin3
        ds.attrs["ipob2"] = ipob2
        ds.attrs["ipob3"] = ipob3
        ds.attrs["ndp2"] = ndp2
        ds.attrs["ndp3"] = ndp3
        ds.attrs["iparam"] = slf.header.params
        ds.attrs["var_IDs"] = slf.header.var_IDs
        ds.attrs["varnames"] = vars
        ds.attrs["varunits"] = slf.header.var_units
        ds.attrs["date"] = slf.header.date
        if nplan > 1:
            # Adding additional metadata as attributes
            ds.attrs["type"] = "3D"
        else:
            ds.attrs["type"] = "2D"

        return ds

    @staticmethod
    def guess_can_open(filename_or_obj):
        try:
            _, ext = os.path.splitext(str(filename_or_obj))
        except TypeError:
            return False
        return ext.lower() in {".slf"}

    description = "A SELAFIN file format backend for Xarray"
    url = "https://www.example.com/selafin_backend_documentation"


@xr.register_dataset_accessor("selafin")
class SelafinAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def write(self, filepath, **kwargs):
        """
        Write data from an Xarray dataset to a SELAFIN file.
        Parameters:
        - filename: String with the path to the output SELAFIN file.
        """
        # Assuming ds is your Xarray Dataset
        ds = self._obj

        # Simplified example of writing logic (details need to be implemented):

        write_serafin(filepath, ds)
