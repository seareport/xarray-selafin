from setuptools import setup, find_packages

setup(
    name="xarray-selafin-backend",
    version="0.1.0",
    author="tomsail",
    author_email="saillour.thomas@gmail.com",
    description="https://github.com/seareport/xarray-selafin",
    packages=find_packages(),
    package_data={
        # If your data files are in a package called 'xarray_selafin_backend' under 'data'
        "xarray_selafin_backend": ["data/*", "variable/*"],
    },
    install_requires=[
        "numpy",
        "pytest",
        "scipy",
        "shapely",
        "xarray",
        "netcdf4",
    ],
    extras_require={
        "dask": ["dask"],
        "dev": ["matplotlib"],  # Assuming netcdf4 is listed intentionally in both main and dev dependencies
    },
    entry_points={
        "xarray.backends": [
            "selafin = xarray_selafin_backend.xarray_backend:SelafinBackendEntrypoint",
        ],
    },
    python_requires=">=3.9",
)
