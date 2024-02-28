import os
from datetime import datetime
from datetime import timedelta

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.backends import BackendEntrypoint
from xarray.core import indexing

from xarray_selafin_backend import Serafin


def read_serafin(f):
    resin = Serafin.Read(f, "en")
    resin.__enter__()
    resin.read_header()
    resin.get_time()
    return resin


def write_serafin(fout, ds, file_format):
    slf_header = Serafin.SerafinHeader(
        ds.attrs["title"]
    )  # Avoid changing title to perform comparison
    if file_format == "SERAFIN":
        slf_header.to_single_precision()
    elif file_format == "SERAFIND":
        slf_header.to_double_precision()
    else:
        raise NotImplementedError

    slf_header.endian = ">"

    slf_header.date = ds.attrs["date_start"]

    slf_header.nb_frames = ds.time.size

    for var in ds.data_vars:
        pos = ds.var_IDs.index(var)
        slf_header.var_IDs.append(var)
        slf_header.var_names.append(ds.varnames[pos].ljust(16).encode(Serafin.SLF_EIT))
        slf_header.var_units.append(ds.varunits[pos].ljust(16).encode(Serafin.SLF_EIT))
    slf_header.nb_var = len(slf_header.var_IDs)

    slf_header.params = tuple(ds.attrs["iparam"])
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
    vars = slf_header.var_IDs

    if "plan" in ds.dims:  # 3D
        slf_header.nb_planes = len(ds.plan)
        if slf_header.nb_planes == 1:
            slf_header.nb_planes = 0
        slf_header.is_2d = False
    else:  # 2D (converted if required)
        slf_header.nb_planes = ds.attrs["nplan"]
        if ds.attrs["type"] == "3D":
            slf_header.is_2d = False  # to enable convertion from 3D
            slf_header = slf_header.copy_as_2d()
        slf_header.is_2d = True

    resout = Serafin.Write(fout, "en", overwrite=True)
    resout.__enter__()
    resout.write_header(slf_header)
    shape = (slf_header.nb_var, slf_header.nb_nodes_2d, max(1, slf_header.nb_planes))

    t0 = np.datetime64(datetime(*slf_header.date))

    if slf_header.nb_frames == 1:
        time_serie = [float(0.0)]
    else:
        time_serie = [
            (t.values - t0).astype("timedelta64[s]").astype(int) for t in ds.time
        ]
    for it, t_ in enumerate(time_serie):
        temp = np.empty(shape, dtype=slf_header.np_float_type)
        for iv, var in enumerate(vars):
            if slf_header.nb_frames == 1:
                temp[iv, :] = ds[var]
            else:
                temp[iv, :] = ds.isel(time=it)[var]
        resout.write_entire_frame(
            slf_header,
            t_,
            np.reshape(temp, (slf_header.nb_var, slf_header.nb_nodes)),
        )


class SelafinLazyArray(BackendArray):
    # not implemented yet: too slow
    def __init__(self, slf_reader, var, dtype, shape):
        self.slf_reader = slf_reader
        self.var = var
        self.dtype = dtype
        self.shape = shape

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            time_key, node_key, plan_key = key
        else:
            raise NotImplementedError

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
            raise ValueError("node_key must be an integer or slice")

        if isinstance(plan_key, slice):
            plan_indices = range(*plan_key.indices(self.shape[2]))
        elif isinstance(plan_key, int):
            plan_indices = [plan_key]
        else:
            raise ValueError("plan_key must be an integer or slice")

        # Initialize data array to hold the result
        data_shape = (len(time_indices), len(node_indices), len(plan_indices))
        data = np.empty(data_shape, dtype=self.dtype)

        # Iterate over the time indices to read the required time steps
        for it, t in enumerate(time_indices):
            temp = self.slf_reader.read_var_in_frame(t, self.var)  # shape = (nb_nodes,)
            temp = np.reshape(temp, self.shape[1:])  # shape = (nb_nodes_2d, nb_planes)
            if node_key == slice(None) and plan_key == slice(
                None
            ):  # speedup if not selection
                data[it] = temp
            else:
                data[it] = temp[node_indices][:, plan_indices]

        # Remove dimension if key was an integer
        if isinstance(node_key, int):
            data = data[:, 0, :]
        if isinstance(plan_key, int):
            data = data[..., 0]
        if isinstance(time_key, int):
            data = data[0, ...]
        return data


class SelafinBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
        lazy_loading=True,
    ):
        # Initialize SELAFIN reader
        slf = read_serafin(filename_or_obj)

        # Prepare dimensions, coordinates, and data variables
        times = [datetime(*slf.header.date) + timedelta(seconds=t) for t in slf.time]
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
        vars = slf.header.var_IDs

        # Create data variables
        data_vars = {}
        dtype = np.float64
        shape = (len(times), npoin2, nplan)

        for var in vars:
            if lazy_loading:
                lazy_array = SelafinLazyArray(slf, var, dtype, shape)
                data = indexing.LazilyIndexedArray(lazy_array)
                data_vars[var] = xr.Variable(dims=["time", "node", "plan"], data=data)
            else:
                data = np.empty(shape, dtype=dtype)
                for time_index, t in enumerate(times):
                    data[time_index, :, :] = slf.read_var_in_frame(
                        time_index, var
                    ).reshape(npoin2, nplan)
                data_vars[var] = xr.Variable(dims=["time", "node", "plan"], data=data)

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

        ds.attrs["title"] = slf.header.title.decode(Serafin.SLF_EIT).strip()
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
        ds.attrs["var_IDs"] = vars
        ds.attrs["varnames"] = [
            b.decode(Serafin.SLF_EIT).rstrip() for b in slf.header.var_names
        ]
        ds.attrs["varunits"] = [
            b.decode(Serafin.SLF_EIT).rstrip() for b in slf.header.var_units
        ]
        ds.attrs["date_start"] = slf.header.date
        ds.attrs["nplan"] = slf.header.nb_planes
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

    def write(self, filepath, file_format="SERAFIND", **kwargs):
        """
        Write data from an Xarray dataset to a SELAFIN file.
        Parameters:
        - filename: String with the path to the output SELAFIN file.
        """
        # Assuming ds is your Xarray Dataset
        ds = self._obj

        # Simplified example of writing logic (details need to be implemented):

        write_serafin(filepath, ds, file_format)
