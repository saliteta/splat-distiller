from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any
import torch
from pathlib import Path
import math
import torch.nn.functional as F
import os
import faiss
from plyfile import PlyData
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def points_in_frustum(
    pts_world: torch.Tensor,
    cam_to_world: torch.Tensor,
    K: torch.Tensor,
    img_size: tuple[int, int],
    near: float = 1e-3,
    far: float = 1e3,
) -> torch.Tensor:
    """
    Args:
      pts_world: (N,3)  world‐coordinates
      cam_to_world: (4,4) camera→world homogeneous matrix
      K: (3,3)           intrinsic matrix
      img_size: (W, H)   image width and height in pixels
      near, far:         clipping planes along Z_cam

    Returns:
      mask: (N,) boolean, True if point is inside the frustum.
    """
    N = pts_world.shape[0]
    device = pts_world.device

    # 1) world→camera
    world_to_cam = torch.inverse(cam_to_world.to(device))  # (4,4)
    pts_h = torch.cat([pts_world, torch.ones(N, 1, device=device)], 1)  # (N,4)
    pts_cam_h = (world_to_cam @ pts_h.T).T  # (N,4)
    pts_cam = pts_cam_h[:, :3]  # (N,3)
    Xc, Yc, Zc = pts_cam.unbind(1)

    # 2) depth‐mask: in front and within [near,far]
    depth_mask = (Zc > near) & (Zc < far)

    # 3) project to pixel coords
    #    in homogeneous: [u·Z; v·Z; Z] = K @ [Xc; Yc; Zc]
    proj = (K.to(device) @ pts_cam.T).T  # (N,3)
    u = proj[:, 0] / proj[:, 2]
    v = proj[:, 1] / proj[:, 2]

    # 4) image‐bounds mask
    W, H = img_size
    bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 5) combine
    return depth_mask & bounds_mask


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
    def geometry(self) -> Dict[str, torch.Tensor | int]:
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
    def to(self, device: torch.device | str) -> None:
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
        self._source_data = {}
    
    def from_file(self, file_path: Union[str, Path], feature_path: Optional[Union[str, Path]] = None, args: Optional[dict] = None, tikhonov: Union[float, None] = None) -> None:
        if file_path.endswith(".ply"):
            self._load_ply(file_path, tikhonov=tikhonov, args=args)
        elif file_path.endswith(".pt"):
            self._from_ckpt(file_path, tikhonov=tikhonov, args=args)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        if feature_path is not None:
            self._feature = torch.load(feature_path, map_location="cuda")
        else:
            possible_feature_path = Path(file_path).parent / (
                Path(file_path).stem + "_features.pt"
            )
            if os.path.exists(possible_feature_path):
                self._feature = torch.load(possible_feature_path, map_location="cuda")
            else:
                self._feature = None


    def _from_ckpt(
        self,
        file_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
        tikhonov: Union[float, None] = None,
        args: Optional[dict] = None,
    ) -> None:
        ckpt = torch.load(file_path, map_location="cuda")["splats"]

        self._source_data["means"] = ckpt["means"].cpu()
        self._source_data["quats"] = ckpt["quats"].cpu()
        self._source_data["scales"] = ckpt["scales"].cpu()
        self._source_data["opacities"] = ckpt["opacities"].cpu()
        self._source_data["sh0"] = ckpt["sh0"].cpu()
        self._source_data["shN"] = ckpt["shN"].cpu()

        means = ckpt["means"]
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)
        scales = torch.exp(ckpt["scales"])
        if tikhonov is not None:
            opacities = torch.sigmoid(ckpt["opacities"] * tikhonov)
        else:
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
            self._source_data["features"] = features.cpu()
        else:
            print("No features found, using random features")
            self._feature = None
        print("Number of Gaussians:", len(means))

    def _load_ply(self, path, use_train_test_exp = False, tikhonov: Union[float, None] = None, args: Optional[dict] = None):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        mask = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        means = torch.tensor(xyz, dtype=torch.float, device="cuda")
        quats = torch.tensor(rots, dtype=torch.float, device="cuda")
        scales = torch.tensor(scales, dtype=torch.float, device="cuda")
        opacities = torch.tensor(opacities, dtype=torch.float, device="cuda")
        sh_degree = 2
        sh0 = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        shn = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()

        self._source_data["means"] = means.cpu()
        self._source_data["quats"] = quats.cpu()
        self._source_data["scales"] = scales.cpu()
        self._source_data["opacities"] = opacities.cpu()
        self._source_data["sh0"] = sh0.cpu()
        self._source_data["shN"] = shn.cpu()

        quats = F.normalize(quats, p=2, dim=-1)
        scales = torch.exp(scales)
        if tikhonov is not None:
            opacities = torch.sigmoid(opacities*tikhonov)
        else:
            opacities = torch.sigmoid(opacities)
        self._geometry = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities.squeeze(-1),
            "sh_degree": 2,
        }

        self._color = {"colors": torch.cat([sh0, shn], dim=-2)}
        self.active_sh_degree = 2

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
            if isinstance(value, torch.Tensor):
                output.append(f"{key}: {value.shape}")
            else:
                output.append(f"{key}: {value}")
        for key, value in self.color.items():
            if isinstance(value, torch.Tensor):
                output.append(f"{key}: {value.shape}")
            else:
                output.append(f"{key}: {value}")
        if self.feature is not None:
            output.append(f"feature: {self.feature.shape}")
        return "\n".join(output)

    def to(self, device: torch.device | str) -> None:
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
        dict_to_save["splats"] = self._source_data
        torch.save(dict_to_save, file_path)

    def mask(self, mask: torch.Tensor) -> None:
        """
        Mask the primitive., given positive mask
        That means the splats that are not in the mask will be removed.
        """
        for key, value in self._geometry.items():
            if isinstance(value, torch.Tensor):
                self._geometry[key] = value[mask]
        for key, value in self.color.items():
            if isinstance(value, torch.Tensor):
                self._color[key] = value[mask]
        if self._feature is not None:
            self._feature = self._feature[mask]
        mask = mask.cpu()
        for key, value in self._source_data.items():
            if isinstance(value, torch.Tensor):
                self._source_data[key] = value[mask]

    def filtering(
        self,
        trainloader: torch.utils.data.DataLoader,
        means: torch.Tensor,
        device: torch.device,
        threshold: float = 0.2,
        args: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Filtering the splats that is not in the center of the image
        This is a simple way to filter the splats that is not in the center of the image
        Although it is very trivial, but it align with the user request for reconstructing
        objects and fast segmentation. For example, fast 3D assets generation.
        when calling filering, it will automatically generate a mask for the splats, and apply the masks to the primitive

        We are not using this function for evaluation, but for real world application.


        Args:
            trainloader: the dataloader for training
            means: the means of the splats
            device: the device to use
            threshold: the threshold for filtering

        Returns:
            splat_mask: the mask for the splats
        """
        splat_weights = torch.zeros(
            means.shape[0], 1, device=device
        )  # all splats are filtered

        for data in tqdm(trainloader, desc="Filtering splats", total=len(trainloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            height, width = data["image"].shape[1:3]

            splat_weights += points_in_frustum(
                means, camtoworlds, Ks, (width, height), near=1e-3, far=1e3
            ).to(torch.float32)
        splat_mask = torch.topk(
            splat_weights, k=int(splat_weights.shape[0] * threshold), dim=0
        )[1]
        self.mask(splat_mask)
        return splat_mask.squeeze(-1)


class DrSplatPrimitive(GaussianPrimitive):
    """
    A DrSplat primitive is a 3D Gaussian distribution.
    """

    def __init__(self) -> None:
        super().__init__()

    def from_file(
        self,
        file_path: Union[str, Path],
        faiss_index_path: Optional[Union[str, Path]] = None,
    ) -> None:
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
        assert (
            len(ckpt) == 13 or 12
        ), f"13 means with feature 12 means no features, you have {len(ckpt)}"
        if len(ckpt) == 13:
            assert (
                faiss_index_path is not None
            ), "faiss_index_path is required when 13 means with feature"
            print("13 means with feature, loading feature")
            (
                sh_degree,
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity,
                features,
                _,
                _,
                _,
                _,
                _,
            ) = ckpt
            zero_masks = torch.all(features == -1, dim=-1)
            self._faiss_index = faiss.read_index(faiss_index_path)
            valid_feature = torch.from_numpy(
                self._faiss_index.sa_decode(features[~zero_masks].cpu().numpy())
            )
            self._feature = torch.zeros(len(means), 512)
            self._feature[~zero_masks] = valid_feature
        else:
            print("12 means no feature, loading no feature")
            (
                sh_degree,
                means,
                sh0,
                shN,
                scaling,
                rotation,
                opacity,
                _,
                _,
                _,
                _,
                _,
            ) = ckpt
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


class GaussianPrimitive2D(GaussianPrimitive):
    """
    A Gaussian primitive is a 3D Gaussian distribution.
    """

    def __init__(self) -> None:
        super().__init__()


class BetaSplatPrimitive(GaussianPrimitive):
    """
    A BetaSplat primitive is a 3D Beta distribution.
    """

    def __init__(self) -> None:
        super().__init__()

    def _sb_params_activation(self, sb_params):
        softplus_sb_params = F.softplus(sb_params[..., :3], beta=math.log(2) * 10)
        sb_params = torch.cat([softplus_sb_params, sb_params[..., 3:]], dim=-1)
        return sb_params

    def _from_dbs_ckpt(
        self,
        ckpt_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load a BetaSplat primitive from a checkpoint.
        """
        ckpt, _ = torch.load(ckpt_path, map_location="cuda")
        (
            sh_degree,
            means,
            sh0,
            shN,
            sb_params,
            scaling,
            rotation,
            opacity,
            beta,
            _,
            _,
        ) = ckpt
        self._geometry = {
            "means": means,
            "opacities": torch.sigmoid(opacity.squeeze(-1)),
            "scales": torch.exp(scaling),
            "quats": F.normalize(rotation),
            "sh_degree": sh_degree,
            "sb_number": sb_params.shape[1],
        }
        self._color = {
            "sh0": sh0,
            "sb_params": self._sb_params_activation(sb_params),
            "beta": self._beta_activation(beta.squeeze(-1)),
        }
        self._load_feature_from_ply(ckpt_path, feature_path)

    def _beta_activation(self, betas):
        return 4.0 * torch.exp(betas)

    def _from_ply(
        self,
        file_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load a BetaSplat primitive from a file.
        """
        plydata = PlyData.read(file_path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        betas = np.asarray(plydata.elements[0]["beta"])[..., np.newaxis]

        sh0 = np.zeros((xyz.shape[0], 3, 1))
        sh0[:, 0, 0] = np.asarray(plydata.elements[0]["sh0_0"])
        sh0[:, 1, 0] = np.asarray(plydata.elements[0]["sh0_1"])
        sh0[:, 2, 0] = np.asarray(plydata.elements[0]["sh0_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("shN_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 0

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("sb_params_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 2 * 6
        sb_params = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            sb_params[:, idx] = np.asarray(plydata.elements[0][attr_name])
        sb_params = sb_params.reshape((sb_params.shape[0], 6, 2))

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._source_data = {
            "means": torch.tensor(xyz, dtype=torch.float, device="cuda").cpu(),
            "sh0": torch.tensor(sh0, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .cpu(),
            "sb_params": torch.tensor(sb_params, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .cpu(),
            "opacity": torch.tensor(opacities, dtype=torch.float, device="cuda").cpu(),
            "beta": torch.tensor(betas, dtype=torch.float, device="cuda").cpu(),
            "scaling": torch.tensor(scales, dtype=torch.float, device="cuda").cpu(),
            "rotation": torch.tensor(rots, dtype=torch.float, device="cuda").cpu(),
        }

        means = torch.tensor(xyz, dtype=torch.float, device="cuda")
        sh0 = torch.tensor(sh0, dtype=torch.float, device="cuda").transpose(1, 2)
        sb_params = torch.tensor(sb_params, dtype=torch.float, device="cuda").transpose(
            1, 2
        )
        opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        beta = torch.tensor(betas, dtype=torch.float, device="cuda")
        scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        self._geometry = {
            "means": means,
            "opacities": torch.sigmoid(opacity.squeeze(-1)),
            "scales": torch.exp(scaling),
            "quats": F.normalize(rotation),
            "sh_degree": 0,
            "sb_number": 2,
        }
        self._color = {
            "sh0": sh0,
            "sb_params": self._sb_params_activation(sb_params),
            "beta": self._beta_activation(beta.squeeze(-1)),
        }
        self._load_feature_from_ply(file_path, feature_path)

    def _load_feature_from_ply(
        self, ckpt_path: Union[str, Path], feature_path: Union[str, Path] | None = None
    ) -> None:
        """
        Load features from a ply file.
        """
        if feature_path is not None:
            self._feature = torch.load(feature_path, map_location="cuda")
        elif os.path.exists(
            Path(ckpt_path).parent / (Path(ckpt_path).stem + "_features.pt")
        ):
            possible_feature_path = Path(ckpt_path).parent / (
                Path(ckpt_path).stem + "_features.pt"
            )
            print(f"Using features from {possible_feature_path}")
            self._feature = torch.load(possible_feature_path, map_location="cuda")
        else:
            self._feature = None
            print("No features found")
        if self._feature is not None:
            self._source_data["features"] = self._feature.cpu()

    def _from_ours_ckpt(
        self,
        ckpt_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
        transformed: bool = False,
    ) -> None:
        """
        Load a BetaSplat primitive from a checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location="cuda")["splats"]
        means = ckpt["means"]
        sh0 = ckpt["sh0"]
        sb_params = ckpt["sb_params"]
        scaling = ckpt["scaling"]
        rotation = ckpt["rotation"]
        opacity = ckpt["opacity"]
        betas = ckpt["beta"]
        self._source_data = {
            "means": means,
            "sh0": sh0,
            "sb_params": sb_params,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "beta": betas,
        }
        self._geometry = {
            "means": means,
            "opacities": torch.sigmoid(opacity.squeeze(-1)),
            "scales": torch.exp(scaling),
            "quats": F.normalize(rotation),
            "sh_degree": 0,
            "sb_number": sb_params.shape[1],
        }
        self._color = {
            "sh0": sh0,
            "sb_params": self._sb_params_activation(sb_params),
            "beta": self._beta_activation(betas.squeeze(-1)),
        }
        self._load_feature_from_ply(ckpt_path, feature_path)

    def from_file(
        self,
        file_path: Union[str, Path],
        feature_path: Optional[Union[str, Path]] = None,
        transformed: bool = False,
    ) -> None:
        """
        Load a BetaSplat primitive from a file.
        """
        if Path(file_path).suffix == ".ply":
            self._from_ply(file_path, feature_path)
        elif Path(file_path).suffix == ".pt":
            try:
                self._from_ours_ckpt(file_path, feature_path, transformed)
            except Exception as e:
                self._from_dbs_ckpt(file_path, feature_path)
        else:
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

    def filtering(
        self,
        trainloader: torch.utils.data.DataLoader,
        means: torch.Tensor,
        device: torch.device,
        threshold: float = 0.2,
        args: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Filtering the splats that is not in the center of the image
        This is a simple way to filter the splats that is not in the center of the image
        Although it is very trivial, but it align with the user request for reconstructing
        objects and fast segmentation. For example, fast 3D assets generation.
        """

        if args is None:
            return super().filtering(trainloader, means, device, threshold)

        if "space_filter" in args.keys():
            super().filtering(trainloader, means, device, threshold)

        if "beta_filter_small" in args.keys():
            threshold = args["beta_filter_small"]
            mask = self.color["beta"] > threshold
            self.mask(mask)

        if "beta_filter_large" in args.keys():
            threshold = args["beta_filter_large"]
            mask = self.color["beta"] < threshold
            self.mask(mask)

        return self._geometry["means"]


__all__ = [
    "BetaSplatPrimitive",
    "GaussianPrimitive",
    "GaussianPrimitive2D",
    "DrSplatPrimitive",
    "Primitive",
]
