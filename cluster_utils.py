import torch
import numpy as np
import cupy as cp
from cuml.cluster import HDBSCAN
from typing import Tuple


def cluster_1d_cuml(
    data: torch.Tensor,
    min_cluster_size: int = 1,
    min_samples: int = None,
    metric: str = "euclidean",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cluster 1-D data using GPU-accelerated cuML HDBSCAN and return:
      - labels:        (n,) LongTensor of cluster IDs (-1 = noise)
      - prototypes:    (k,) FloatTensor of cluster means for clusters 0..k-1
      - assigned:      (n,) FloatTensor where each entry is the prototype of its cluster
    """
    # 1) to CPU numpy and reshape to (n,1)
    X = data.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
    n = X.shape[0]

    # 2) move to GPU
    X_gpu = cp.asarray(X)

    # 3) run cuML HDBSCAN
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        metric=metric,
    )
    labels_gpu = hdb.fit_predict(X_gpu)  # CuPy array shape (n,)
    labels = cp.asnumpy(labels_gpu).astype(np.int32)  # (n,)

    # 4) compute prototypes (mean of points per cluster)
    unique_labels = np.unique(labels[labels >= 0])
    if unique_labels.size > 0:
        max_lab = int(unique_labels.max())
        prototypes = np.full((max_lab + 1,), np.nan, dtype=np.float32)
        for lab in unique_labels:
            prototypes[lab] = X[labels == lab].mean()
    else:
        prototypes = np.array([], dtype=np.float32)

    # 5) assign each sample its cluster's prototype (noise → NaN)
    assigned = np.array(
        [prototypes[lab] if lab >= 0 else np.nan for lab in labels], dtype=np.float32
    )

    return (
        torch.from_numpy(labels).long(),
        torch.from_numpy(prototypes).float(),
        torch.from_numpy(assigned).float(),
    )
