# instant_lerf/setup.py


import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name="splat_distiller",
    version="0.1.0",
    install_requires=[
        # Git dependencies
        "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
        "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157",
        "FeatUp @ git+https://github.com/mhamilton723/FeatUp.git@main",
        "CLIP @ git+https://github.com/mhamilton723/CLIP.git@main",
        
        # PyPI dependencies
        "viser",
        "nerfview",
        "imageio[ffmpeg]",
        "numpy",
        "scikit-learn",
        "tqdm",
        "torchmetrics[image]",
        "opencv-python",
        "tyro>=0.8.8",
        "Pillow",
        "tensorboard",
        "tensorly",
        "pyyaml",
        "matplotlib",
        "plyfile",
        "setuptools==72.1.0"
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)