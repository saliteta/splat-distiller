import os
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
import subprocess
import sys

install_requires = [
    # Git dependencies
    "pycolmap @ git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e",
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157",
    "FeatUp @ git+https://github.com/opipari/FeatUp.git@main",
    "CLIP @ git+https://github.com/mhamilton723/CLIP.git@main",
    # PyPI dependencies
    "viser",
    "open_clip_torch",
    "nerfview @ git+https://github.com/RongLiu-Leo/nerfview.git",
    "imageio[ffmpeg]",
    "ninja",
    "numpy<2.0.0",
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
    "setuptools==72.1.0",
    "jaxtyping",
    "rich>=12",
    "torch",
    "typing_extensions; python_version<'3.8'",
    "splines",
    "plas @ git+https://github.com/fraunhoferhhi/PLAS.git",
    "pandas",
    "tabulate",
    "black[jupyter]==22.3.0",
    "hydra-core",
    "omegaconf",
]


class BuildSubmodule(Command):
    """Custom command to install the submodule located in the 'submodule' folder."""

    description = "Install the submodule package from the 'submodule' folder."
    user_options = []  # No options needed for this command

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        submodule_path = os.path.join(os.path.dirname(__file__), "submodules")
        # Use pip to install the submodule; adjust the arguments as needed.
        subprocess.check_call(
            ["pip", "install", "."], cwd=os.path.join(submodule_path, "bsplat")
        )
        subprocess.check_call(
            ["pip", "install", "."], cwd=os.path.join(submodule_path, "gsplat")
        )
        subprocess.check_call(
            ["pip", "install", "."], cwd=os.path.join(submodule_path, "gsplat_ext")
        )
        subprocess.check_call(
            ["pip", "install", "."], cwd=os.path.join(submodule_path, "segment-anything-langsplat")
        )


class CustomInstall(install):
    """Custom install command that installs the submodule before installing the main project."""

    def run(self):
        # First, install the submodule
        self.run_command("build_submodule")
        # Then proceed with the standard installation of the main project
        install.run(self)



setup(
    name="splat_distiller",
    version="0.1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={
        "build_submodule": BuildSubmodule,
        "install": CustomInstall,
    },
)
