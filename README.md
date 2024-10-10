# PyL2BV
 Next implementation of the BioRetrieval PyL2BV program,
 which stands for Python Level 2B Vegetation.

# Installation Guide

## Prerequisites

Ensure you have Python 3.10.12 installed on your system.

## Creating a Virtual Environment

### Using `venv`

1. Create a virtual environment:
    ```bash
    python3 -m venv myenv
    ```
2. Activate the virtual environment:
    - On macOS/Linux:
        ```bash
        source myenv/bin/activate
        ```
    - On Windows:
        ```powershell
        .\myenv\Scripts\activate
        ```

### Using `conda`

1. Create a new conda environment:
    ```bash
    conda create --name myenv python=3.10.12
    ```
2. Activate the conda environment:
    ```bash
    conda activate myenv
    ```

## Installing Packages

Once you have activated your virtual environment, install the necessary packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
