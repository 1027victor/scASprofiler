from __future__ import annotations

import numpy as np


def knn_observed(
    data: np.ndarray,
    mask: np.ndarray,
    generate_data: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """KNN-impute within each cell using simulated samples of the same class.

    Parameters
    ----------
    data
        1D vector of length n_features, with zeros where missing.
    mask
        1D binary mask (1=observed, 0=missing) for the *original* data.
    generate_data
        Array of shape (sim_size, n_features) representing generated samples.
    """
    sim_size = generate_data.shape[0]
    feat_len = data.shape[0]

    sim_flat = generate_data.reshape(sim_size, feat_len).T  # (feat_len, sim_size)
    obs_mask = mask.astype(bool)

    # Only compute distances on observed positions
    diff = np.repeat(data[:, None], sim_size, axis=1) - sim_flat
    dists = diff * diff
    dists[~obs_mask, :] = np.nan

    neigh_idx = np.nanmean(dists, axis=0).argsort()[:k]
    sim_med = np.nanmedian(sim_flat[:, neigh_idx], axis=1)

    out = data.copy()
    out[~obs_mask] = sim_med[~obs_mask]
    return out

