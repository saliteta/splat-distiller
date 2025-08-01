import torch
import numpy as np
import math
from sklearn.decomposition import PCA, IncrementalPCA
from typing import Tuple
from cuml.cluster import HDBSCAN
import cupy as cp
import torch.nn.functional as F
import os

def positional_embedding(
    positions: torch.Tensor,
    num_freqs: int = 4
) -> torch.Tensor:
    """
    positions: (N,3) tensor with coords in [-1,1]
    num_freqs: number of frequency bands per axis;
               total output dim = 6 * num_freqs
    returns: (N, 6*num_freqs) positional embedding
    """
    # unpack
    x, y, z = positions.unbind(-1)  # each (N,)

    # build freq bands: 1,2,4,...
    freq_bands = 2.0 ** torch.arange(num_freqs, device=positions.device, dtype=positions.dtype)  # (num_freqs,)

    # compute angles: pos * freq * π
    ang_x = x.unsqueeze(-1) * freq_bands * math.pi  # (N, num_freqs)
    ang_y = y.unsqueeze(-1) * freq_bands * math.pi
    ang_z = z.unsqueeze(-1) * freq_bands * math.pi

    # sin / cos
    pe_x = torch.cat([ang_x.sin(), ang_x.cos()], dim=-1)  # (N,2*num_freqs)
    pe_y = torch.cat([ang_y.sin(), ang_y.cos()], dim=-1)
    pe_z = torch.cat([ang_z.sin(), ang_z.cos()], dim=-1)

    # concat
    return torch.cat([pe_x, pe_y, pe_z], dim=-1)        # (N,6*num_freqs)


def fast_hdbscan_with_pos_gpu(
    embeddings: torch.Tensor,
    positions: torch.Tensor|None = None,
    num_freqs: int = 3,
    n_components: int = 50,
    min_cluster_size: int = 500,
    min_samples: int = None,
    use_incremental_pca: bool = False,
    ipca_batch_size: int = 20000,
    positional_weight: float = 1e-2,
):
    """
    GPU‐accelerated PCA (on CPU) + cuML HDBSCAN (on GPU).
    Returns:
      labels:        numpy.ndarray (N,) int32
      probabilities: numpy.ndarray (N,) float32
      pca_feat:      the fitted sklearn PCA (or IncrementalPCA)
      pca_pos:       PCA for positions (or None)
    """
    # 1) feature PCA on CPU
    X = embeddings.detach().cpu().numpy()
    if use_incremental_pca:
        pca_feat = IncrementalPCA(n_components=n_components)
        for i in range(0, X.shape[0], ipca_batch_size):
            pca_feat.partial_fit(X[i:i+ipca_batch_size])
        X_red = np.vstack([
            pca_feat.transform(X[i:i+ipca_batch_size])
            for i in range(0, X.shape[0], ipca_batch_size)
        ])
    else:
        pca_feat = PCA(n_components=n_components, random_state=0)
        X_red = pca_feat.fit_transform(X)  # (N, n_components)

    # 2) optional positional embed on CPU
    if positions is not None:
        # align via PCA, normalize to [-1,1]
        pos_np = positions.detach().cpu().numpy()
        pca_pos = PCA(n_components=3, random_state=0)
        pos_aligned = pca_pos.fit_transform(pos_np)
        mins, maxs = pos_aligned.min(0), pos_aligned.max(0)
        pos_norm = 2 * (pos_aligned - mins) / (maxs - mins + 1e-8) - 1
        # get PE
        pos_t = torch.from_numpy(pos_norm).float()
        pe = positional_embedding(pos_t, num_freqs=num_freqs).numpy()
        # weight & concat
        data_cpu = np.hstack([X_red, pe * positional_weight])
    else:
        pca_pos = None
        data_cpu = X_red

    # 3) move to GPU
    data_gpu = cp.asarray(data_cpu, dtype=cp.float32)
    assert not cp.isnan(data_gpu).any()
    assert not cp.isinf(data_gpu).any()

    # 4) cuML HDBSCAN
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'  # or 'cosine' if supported on GPU
    )
    labels_gpu = hdb.fit_predict(data_gpu)
    # probabilities_ only available if prediction_data=True, omitted for speed

    # 5) bring back to CPU
    labels = cp.asnumpy(labels_gpu).astype(np.int32)
    # if you enabled prediction_data:
    # probs = cp.asnumpy(hdb.probabilities_).astype(np.float32)
    probs = None

    return labels, probs, pca_feat, pca_pos
    
def compute_prototypes(
    labels: torch.Tensor,
    features: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cluster prototypes and assign each sample its prototype feature.

    Args:
        labels: (n,) int Tensor of cluster labels (may include -1 for noise)
        features: (n, d) float Tensor of original features

    Returns:
        unique_labels: (k,) Tensor of sorted unique labels
        prototypes: (k, d) Tensor of mean feature per label
        assigned_features: (n, d) Tensor where each row is the prototype of its label
    """
    device = features.device
    dtype = features.dtype
    n, d = features.shape

    # 1) get sorted unique labels
    unique_labels = torch.unique(labels)

    # 2) compute prototype (mean) for each label
    prototypes = torch.stack([
        features[labels == lab].mean(dim=0)
        if (labels == lab).any()
        else torch.zeros(d, device=device, dtype=dtype)
        for lab in unique_labels
    ], dim=0)  # shape (k, d)

    prototypes = prototypes/prototypes.norm(dim=-1, keepdim=True)

    assigned_features = prototypes[labels+1]

    return prototypes, assigned_features
