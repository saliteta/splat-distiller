from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any
import torch
from pathlib import Path
import math
import torch.nn.functional as F
import os


class Primitive(ABC):
    """Base class for 3D primitives.

    This class defines the interface for primitives that must have:
    - geometry: Dictionary containing geometric properties (e.g., means, scales)
    - color: Dictionary containing color properties (e.g., RGB values)
    - feature: Optional tensor containing feature vectors
    """

    @abstractmethod
    def from_file(self, file_path: Union[str, Path]) -> None:
        """Load a primitive from a file.
        This will be implemented in the subclass. It should set the geometry or feature data of the primitive.
        Args:
            file_path: Path to the file containing primitive data
        Returns:
            None
        """
        pass

    @property
    @abstractmethod
    def geometry(self) -> Dict[str, torch.Tensor]:
        """Get the geometry data of the primitive."""
        pass

    @property
    @abstractmethod
    def color(self) -> Dict[str, torch.Tensor]:
        """Get the color data of the primitive."""
        pass

    @property
    @abstractmethod
    def feature(self) -> Optional[torch.Tensor]:
        """Get the feature data of the primitive."""
        pass

    @abstractmethod
    def verbose(self) -> str:
        """Get a verbose string representation of the primitive."""
        output = []
        for key, value in self.geometry.items():
            output.append(f"{key}: {value.shape}")
        for key, value in self.color.items():
            if key != "sh_degree":
                output.append(f"{key}: {value.shape}")
        if self.feature is not None:
            output.append(f"feature: {self.feature.shape}")
        return "\n".join(output)


class GaussianPrimitive(Primitive):
    """
    A Gaussian primitive is a 3D Gaussian distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        self._geometry = {}
        self._color = {}
        self._feature = None

    def from_file(
        self,
        file_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
    ) -> None:
        ckpt = torch.load(file_path, map_location="cuda")["splats"]
        means = ckpt["means"]
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)
        scales = torch.exp(ckpt["scales"])
        opacities = torch.sigmoid(ckpt["opacities"])
        sh0 = ckpt["sh0"]
        shN = ckpt["shN"]
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

        self._geometry = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "sh_degree": sh_degree,
        }
        self._color = {"colors": colors}
        possible_feature_path = Path(file_path).parent / (
            Path(file_path).stem + "_features.pt"
        )
        if feature_path is not None or os.path.exists(possible_feature_path):
            print(
                f"Using features from {feature_path if feature_path is not None else possible_feature_path}"
            )
            features = torch.load(
                feature_path if feature_path is not None else possible_feature_path,
                map_location="cuda",
            )
            self._feature = features
        else:
            print("No features found, using random features")
            self._feature = None
        print("Number of Gaussians:", len(means))

    @property
    def geometry(self) -> Dict[str, torch.Tensor]:
        return self._geometry

    @property
    def color(self) -> Dict[str, torch.Tensor]:
        return self._color

    @property
    def feature(self) -> Optional[torch.Tensor]:
        return self._feature

    def verbose(self) -> str:
        """Get a verbose string representation of the primitive."""
        output = []
        for key, value in self.geometry.items():
            if key != "sh_degree":
                output.append(f"{key}: {value.shape}")
        for key, value in self.color.items():
            output.append(f"{key}: {value.shape}")
        if self.feature is not None:
            output.append(f"feature: {self.feature.shape}")
        return "\n".join(output)
