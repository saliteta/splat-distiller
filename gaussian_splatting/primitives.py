from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any
import torch
from pathlib import Path
import math
import torch.nn.functional as F
import os
import faiss    

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

    @abstractmethod
    def to(self, device: torch.device|str) -> None:
        """Move the primitive to the device."""
        pass


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
    
    def to(self, device: torch.device|str) -> None:
        """Move the primitive to the device."""
        for key, value in self._geometry.items():
            if isinstance(value, torch.Tensor):
                self._geometry[key] = value.to(device)
        for key, value in self.color.items():
            if isinstance(value, torch.Tensor):
                self._color[key] = value.to(device)
        if self._feature is not None:
            self._feature = self._feature.to(device)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the primitive to a file."""
        dict_to_save = {}
        for key, value in self._geometry.items():
            dict_to_save[key] = value
        dict_to_save["sh0"] = self._color["colors"][:, 1, :]
        dict_to_save["shN"] = self._color["colors"][:, 1:, :]
        torch.save(dict_to_save, file_path)



class DrSplatPrimitive(GaussianPrimitive):
    """
    A DrSplat primitive is a 3D Gaussian distribution.
    """

    def __init__(self) -> None:
        super().__init__()

    

    def from_file(self, file_path: Union[str, Path], faiss_index_path: Optional[Union[str, Path]] = None) -> None:
        """
        Notice that in DrSplat, there are some splats are not assign any features,
        we use -1 to mark it as invalid. And during visualization, we make that feature as 0.
        Args:
            file_path: the path to the checkpoint
            faiss_index_path: the path to the faiss index

        Returns:
            None
        """
        ckpt, _ = torch.load(file_path, map_location="cuda", weights_only=False)
        assert len(ckpt) == 13 or 12, f"13 means with feature 12 means no features, you have {len(ckpt)}"
        if len(ckpt) == 13:
            assert faiss_index_path is not None, "faiss_index_path is required when 13 means with feature"
            print("13 means with feature, loading feature")
            (sh_degree, means, sh0, shN, scaling, rotation, opacity,features, _, _, _,_, _) = ckpt
            zero_masks = torch.all(features == -1, dim=-1)
            self._faiss_index = faiss.read_index(faiss_index_path)
            valid_feature = torch.from_numpy(self._faiss_index.sa_decode(features[~zero_masks].cpu().numpy()))
            self._feature = torch.zeros(len(means), 512)
            self._feature[~zero_masks] = valid_feature
        else:
            print("12 means no feature, loading no feature")
            (sh_degree, means, sh0, shN, scaling, rotation, opacity, _, _, _,_, _) = ckpt
            self._feature = None
        colors = torch.cat([sh0, shN], dim=-2)

        self._geometry = {
            "means": means,
            "quats": F.normalize(rotation),
            "scales": torch.exp(scaling),
            "opacities": torch.sigmoid(opacity.squeeze(-1)),
            "sh_degree": sh_degree,
        }
        self._color = {"colors": colors}


    @property
    def geometry(self) -> Dict[str, torch.Tensor]:
        return self._geometry
    
    @property
    def color(self) -> Dict[str, torch.Tensor]:
        return self._color
    
    @property
    def feature(self) -> Optional[torch.Tensor]:
        return self._feature
    

class GaussianPrimitive2D(GaussianPrimitive):
    """
    A Gaussian primitive is a 3D Gaussian distribution.
    """
    def __init__(self) -> None:
        super().__init__()



if __name__ == "__main__":
    primitive = GaussianPrimitive2D()
    primitive.from_file("/media/bxiong/c6deb427-f841-4fb3-8707-2d0593655c63/models/results_metrics/teatime/2DGS/ckpts/ckpt_59999.pt")
    print(primitive.verbose())