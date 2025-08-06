from pathlib import Path
import torch
import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm
import torch.nn.functional as F



"""
    Notice that this pipeline is working only when we do semantic query, and we have original masks
    This script is only for downstream tasks such as semantic segmentation, it should not be used 
    when user has a continuous distributed features to lift. 

    In fact, it only work when dealing with SAM model, since SAM model has many "Semantic Bleeding"
    Artifacts. Furthermore, this method does not provide a good visualization result, but it indeed 
    has a good result when projected back on to 2D map. 

    We get the initial thoughs from LAGA paper, but the explaination about "view-dependent information"
    is not correct. In fact, we think it is a purely denoising procedure that makes the result looks good.
"""

class mask_register:
    def __init__(
        self,
        image_labels: Dict[str, torch.Tensor],
        mask_folder: Path = Path("SAMOpenCLIP_features"),
    ):
        """
        Inputs:
            mask_folder: Path to the folder containing the mask npy files
            label_folder: we changed it to torch tensor dictionary just created
        """
        self.mask_files = list(mask_folder.glob("*s.npy"))
        self.label_names = list(image_labels.keys())
        self.image_labels = image_labels
        self.feature_files = list(mask_folder.glob("*f.npy"))
        self.mask_files.sort()
        self.label_names.sort()
        self.feature_files.sort()
        assert len(self.mask_files) == len(
            self.label_names
        ), f"These two files length must be the same, we got mask_file\
             {len(self.mask_files)}, and label_names {len(self.label_names)}"

    def __len__(self):
        return len(self.mask_files)

    def _register_mask_label(self):
        """
        Construct a mask-to-label mapping: mask determined by frame name + mask ID,
        now recording mIoU between each mask and its most frequent label.
        """
        self.mask_to_label_mapping = {}

        for mask_file, label_file in tqdm(
            zip(self.mask_files, self.label_names),
            total=len(self),
            desc="mask_filtering",
        ):
            self.mask_to_label_mapping[mask_file.name] = {}
            masks = np.load(mask_file)               # (4, H, W)
            labels = self.image_labels[label_file] - 1  # (H, W)
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()

            unique_mask_ids = np.unique(masks[masks != -1])

            for mask_id in unique_mask_ids:
                # boolean mask of pixels belonging to this segment
                mask_pixels = (masks == mask_id).any(axis=0)  # (H, W)

                if not mask_pixels.any():
                    print(f"No pixels for mask {mask_id} in {mask_file.name}")
                    continue

                # look up the ground-truth labels under this mask
                corresponding_labels = labels[mask_pixels]
                unique_labels, counts = np.unique(corresponding_labels, return_counts=True)

                # pick the most frequent label
                idx = np.argmax(counts)
                most_freq_label = unique_labels[idx]

                # build a boolean mask of *all* pixels in the image with that label
                label_pixels = (labels == most_freq_label)

                # compute intersection & union
                intersection = np.logical_and(mask_pixels, label_pixels).sum()
                union        = np.logical_or(mask_pixels, label_pixels).sum()
                miou = intersection / union if union > 0 else 0.0

                # record (label, mIoU)
                self.mask_to_label_mapping[mask_file.name][int(mask_id)] = (
                    int(most_freq_label),
                    float(miou)
                )

        
    def _register_label_features(self, threshold=0.8, debugging = False) -> Dict[int, List[np.ndarray]]:
        """
        Each label will have several discreate mask features
        """
        self.label_to_feature_mapping: Dict[int, List[np.ndarray]] = {}

        for mask_file, feature_file in zip(self.mask_files, self.feature_files):
            features = np.load(feature_file)
            mask2label_dict = self.mask_to_label_mapping[mask_file.name]

            for key, items in mask2label_dict.items():
                cluster_label, trust = items
                if trust > threshold:
                    if cluster_label in self.label_to_feature_mapping.keys():
                        self.label_to_feature_mapping[cluster_label].append(
                            features[int(key)]
                        )
                    else:
                        self.label_to_feature_mapping[cluster_label] = [
                            features[int(key)]
                        ]
        if debugging:
            self._debugging()

        return self.label_to_feature_mapping

    def reg(self):
        self._register_mask_label()
        return self._register_label_features()

    def GaussianFeatureRefinement(
        self,
        continuous_feature: torch.Tensor,
        splat_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Replace features by label‐prototypes where available, but keep
        original continuous features for label == 0 or any label without
        a prototype—all done in one tensorized pass.
        """
        # 1) load
        cont: torch.Tensor = continuous_feature  # (n, c)
        labels: torch.Tensor = splat_labels + 1  # (n,)

        device = cont.device
        dtype = cont.dtype

        # 2) build a prototype lookup table of size (max_label+1, c), init to nan
        max_lbl = int(labels.max().item())
        c = cont.size(1)
        proto_table = torch.full(
            (max_lbl + 1, c), float("nan"), device=device, dtype=dtype
        )

        # 3) fill in real prototypes
        for lbl, feats in self.label_to_feature_mapping.items():
            if lbl == -1:
                continue
            # stack into numpy, mean, convert back
            arr = np.vstack(
                [f.numpy() if isinstance(f, torch.Tensor) else f for f in feats]
            )  # (k, c)
            mean_feat = torch.from_numpy(arr.mean(axis=0)).to(
                device=device, dtype=dtype
            )
            if lbl <= max_lbl:
                proto_table[lbl + 1] = mean_feat

        # 4) index the table by your labels to get per‐sample candidate
        #    shape (n, c), but rows for missing‐labels will be all nan
        cand = proto_table[labels]  # (n, c)

        # 5) build a mask of which rows actually have a valid prototype
        #    (a valid row has no NaNs)
        valid = ~torch.isnan(cand).any(dim=1)  # (n,)

        cont[valid] = cand[valid]
        del cand, proto_table, valid, labels
        norms = cont.norm(2, dim=1, keepdim=True)  # (n,1)
        norms.clamp_min_(1e-6)
        cont.div_(norms)                           # in-place
        del norms
        return cont

    def _debugging(self):
        from cuml.manifold import TSNE
        import matplotlib.pyplot as plt
        import os

        # make sure the output folder exists
        os.makedirs('debugging', exist_ok=True)

        for label, feats in self.label_to_feature_mapping.items():
            if len(feats) == 1:
                continue
            # stack list of (D,) arrays into (N, D)
            X = np.vstack(feats).astype(np.float32)

            # run CUML TSNE
            tsne = TSNE(n_components=2, random_state=42)
            X_2d = tsne.fit_transform(X)

            # plot
            plt.figure(figsize=(6,6))
            plt.scatter(X_2d[:,0], X_2d[:,1], s=5, alpha=0.7)
            plt.title(f"t-SNE for label {label}")
            plt.xlabel("TSNE-1")
            plt.ylabel("TSNE-2")

            # save
            out_path = os.path.join('debugging', f"tsne_label_{label}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved t-SNE plot for label {label} → {out_path}")
