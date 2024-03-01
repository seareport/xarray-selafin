from setuptools import setup, find_packages

setup(
    name="xarray-selafin-backend",
    version="0.1.0",
    author="tomsail",
    author_email="saillour.thomas@gmail.com",
    description="",
    packages=find_packages(),
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
