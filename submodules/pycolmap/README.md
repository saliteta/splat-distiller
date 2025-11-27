# pycolmap
Python interface for COLMAP reconstructions, plus some convenient scripts for
loading/modifying/converting reconstructions.

This code does not, however, run reconstruction -- it only provides a
convenient interface for handling COLMAP's output.

## Installation

The following works with `setuptools >= 62.0.0`. Run `python3 -m pip install
setuptools --upgrade` if necessary.

```
git clone https://github.com/rmbrualla/pycolmap.git
cd pycolmap
python3 -m pip install -e .
```

