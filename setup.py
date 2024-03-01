from setuptools import find_packages
from setuptools import setup

setup(
    name="xarray-selafin",
    version="0.1.2",
    author=["tomsail", "lucduron"],
    author_email="l.duron@cnr.tm.fr",
    description="https://github.com/seareport/xarray-selafin",
    packages=find_packages(),
    package_data={
        "xarray_selafin": ["data/*", "variable/*"],
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
        "dev": [
            "matplotlib"
        ],  # Assuming netcdf4 is listed intentionally in both main and dev dependencies
    },
    entry_points={
        "xarray.backends": [
            "selafin = xarray_selafin.xarray_backend:SelafinBackendEntrypoint",
        ],
    },
    python_requires=">=3.9",
)
