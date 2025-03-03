from setuptools import setup, find_packages

setup(
    name="PyL2BVcli",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "certifi==2024.8.30",
        "cftime==1.6.4.post1",
        "contourpy==1.3.0",
        "cycler==0.12.1",
        "fonttools==4.54.1",
        "joblib==1.4.2",
        "kiwisolver==1.4.7",
        "matplotlib==3.9.2",
        "netCDF4==1.7.2",
        "numpy==2.1.2",
        "packaging==24.1",
        "pillow==11.0.0",
        "pyparsing==3.2.0",
        "pyproj==3.7.0",
        "python-dateutil==2.9.0.post0",
        "scipy==1.14.1",
        "six==1.16.0",
        "spectral==0.23.1",
    ],
    entry_points={
        "console_scripts": [
            "pyl2bv-cli=PyL2BVcli.cli:main",
        ],
    },
)
