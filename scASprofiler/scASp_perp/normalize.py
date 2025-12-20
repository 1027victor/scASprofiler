import numpy as np
from typing import Tuple
import pandas as pd

def up_sample_nan(df: pd.DataFrame, row_num: int) -> pd.DataFrame:
    if df.shape[0] >= row_num:
        return df
    nan_matrix = np.full((row_num - df.shape[0], df.shape[1]), np.nan)
    nan_df = pd.DataFrame(nan_matrix, columns=df.columns)
    return pd.concat([df, nan_df], ignore_index=True)

def make_square_and_normalize(merge_df_numeric: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pad to square (h^2) and normalize by global max (same as your norm_sj = padding / padding.max()).
    """
    sj_coord = merge_df_numeric.shape[0]
    fig_h = int(np.ceil(np.sqrt(sj_coord)))
    padded = up_sample_nan(merge_df_numeric, fig_h**2)
    denom = padded.max().max()
    norm =padded /denom
    return padded, norm

def denorm_sj(merge_df: pd.DataFrame, vae_gan_output_path: str, meta_cols: list) -> pd.DataFrame:
    """
    Denormalize VAE-GAN output back to counts using per-column max from merge_df (filledna 0).
    """
    d = merge_df.fillna(0)
    cell_cols = [c for c in d.columns if c not in set(meta_cols)]
    max_columns = d[cell_cols].max()

    vae = pd.read_csv(vae_gan_output_path, index_col=0)
    vae = vae.iloc[: len(merge_df), : len(cell_cols)]
    vae.index = merge_df.index
    vae.columns = cell_cols

    denorm = vae * max_columns
    out = pd.concat([merge_df[[c for c in meta_cols if c in merge_df.columns]], denorm], axis=1)
    return out