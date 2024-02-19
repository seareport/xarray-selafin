from xarray.backends import BackendEntrypoint
from xarray.backends import BackendArray
from xarray.core import indexing
import numpy as np
import xarray as xr
import os
import logging

from .selafin import Selafin

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


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
        var_idx = self.slf_reader.varnames.index(self.var)
        # Iterate over the time indices to read the required time steps
        for it, t in enumerate(time_indices):
            temp = self.slf_reader.get_values(t)[var_idx]
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
        slf = Selafin(filename_or_obj)

        # Prepare dimensions, coordinates, and data variables
        times = slf.getTimes()
        (
            nelem2,
            nelem3,
            npoin2,
            npoin3,
            ndp2,
            ndp3,
            nplan,
            ikle2,
            ikle3,
            ipob2,
            ipob3,
            x,
            y,
        ) = slf.getMesh()

        # Create data variables using Dask arrays for the variables
        data_vars = {}
        dtype = np.float64
        shape = (len(times), npoin2, nplan)

        if DASK_AVAILABLE:
            for name in slf.varnames:
                lazy_array = SelafinLazyArray(slf, name, dtype, shape)
                dask_array = da.from_array(lazy_array, chunks=(1, shape[1], shape[2]))
                data_vars[name.strip()] = (["time", "node", "plan"], dask_array)

        else:
            for name in slf.varnames:
                var_idx = slf.varnames.index(name)
                data = np.zeros((len(times), npoin2, nplan), dtype=dtype)
                for it, t in enumerate(times):
                    variable_data = slf.get_values(it)[var_idx]
                    data[it, :] = np.reshape(variable_data, (npoin2, nplan))

                data_vars[name.strip()] = (["time", "node", "plan"], data)

        if nplan > 1:
            # Including essential parameters directly in the dataset
            coords = {
                "x": ("node", x),
                "y": ("node", y),
                "time": times,
                # Adding IKLE as a coordinate or data variable for mesh connectivity
                "ikle2": (("nelem2", "ndp2"), ikle2),
                "ikle3": (("nelem3", "ndp3"), ikle3),
                # Consider how to include IPOBO if it's essential for your analysis
            }
        else:
            # Including essential parameters directly in the dataset
            coords = {
                "x": ("node", x),
                "y": ("node", y),
                "time": times,
                # Adding IKLE as a coordinate or data variable for mesh connectivity
                "ikle2": (("nelem2", "ndp2"), ikle2),
                # Consider how to include IPOBO if it's essential for your analysis
            }

        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        ds.attrs["nelem2"] = nelem2
        ds.attrs["nelem3"] = nelem3
        ds.attrs["npoin2"] = npoin2
        ds.attrs["npoin3"] = npoin3
        ds.attrs["ipob2"] = ipob2
        ds.attrs["ipob3"] = ipob3
        ds.attrs["ndp2"] = ndp2
        ds.attrs["ndp3"] = ndp3
        ds.attrs["iparam"] = slf.iparam
        ds.attrs["varnames"] = slf.varnames
        ds.attrs["varunits"] = slf.varunits
        ds.attrs["date"] = slf.datetime
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
        slf_writer = Selafin("")
        slf_writer.setTitle("Converted from Xarray")
        slf_writer.setFile(filepath)
        slf_writer.setMetaData(ds)
        slf_writer.write(ds)
