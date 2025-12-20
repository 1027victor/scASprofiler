from __future__ import annotations

import numpy as np


def my_knn_type(
    data_imp_org_k: np.ndarray,
    mask_k: np.ndarray,
    sim_out_k: np.ndarray,
    knn_k: int = 10,
) -> np.ndarray:
    """KNN-impute within each cell using simulated samples of the same class.

    Parameters
    ----------
    data_imp_org_k
        1D vector of length n_features, with zeros where missing.
    mask_k
        1D binary mask (1=observed, 0=missing) for the *original* data.
    sim_out_k
        Array of shape (sim_size, n_features) representing generated samples.
    """
    sim_size = sim_out_k.shape[0]
    feat_len = data_imp_org_k.shape[0]

    sim_flat = sim_out_k.reshape(sim_size, feat_len).T  # (feat_len, sim_size)
    obs_mask = mask_k.astype(bool)

    # Only compute distances on observed positions
    diff = np.repeat(data_imp_org_k[:, None], sim_size, axis=1) - sim_flat
    dists = diff * diff
    dists[~obs_mask, :] = np.nan

    neigh_idx = np.nanmean(dists, axis=0).argsort()[:knn_k]
    sim_med = np.nanmedian(sim_flat[:, neigh_idx], axis=1)

    out = data_imp_org_k.copy()
    out[~obs_mask] = sim_med[~obs_mask]
    return out

