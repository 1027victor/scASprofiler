# import numpy as np
import pandas as pd

META11 = ["chromosome", "strand", "start", "end", "gene_id", "gene_id_start",
          "gene_id_end", "n_genes", "gene_name", "coordinates", "annotated"]


def _cell_cols_from_repeat(df_repeat: pd.DataFrame):
    if "chromosome" not in df_repeat.columns:
        raise ValueError(
            "Cannot locate 'chromosome' in df_repeat; layout mismatch.")
    meta_start = df_repeat.columns.get_loc("chromosome")
    return df_repeat.columns[1:meta_start].tolist()


def qc_filter_sites(df_repeat: pd.DataFrame, thres: int) -> pd.DataFrame:
    cell_cols = _cell_cols_from_repeat(df_repeat)

    site_count = df_repeat[cell_cols].count(axis=1)

    df_site = df_repeat[cell_cols].copy()
    df_site["site_count"] = site_count.values
    df_site_filtered = df_site[df_site["site_count"] > thres].copy()

    df_site_filtered["coord.intron"] = df_repeat.loc[df_site_filtered.index,
                                                     "coord.intron"].values
    for c in META11:
        if c in df_repeat.columns:
            df_site_filtered.loc[df_site_filtered.index,
                                 c] = df_repeat.loc[df_site_filtered.index, c].values

    return df_site_filtered


def qc_filter_samples(df_site_filtered: pd.DataFrame, thres: int) -> pd.DataFrame:
    sample_count = df_site_filtered.count(axis=0)

    df_sample = df_site_filtered.copy()
    df_sample.loc["sample_count"] = sample_count.values

    keep_cols = df_sample.columns[df_sample.loc["sample_count"] > thres].tolist(
    )
    df_sample_filtered = df_sample.loc[:, keep_cols].drop(
        index=["sample_count"], errors="ignore")

    df_sample_filtered = df_sample_filtered.drop(
        columns=[c for c in ["site_count", "intron_group", "intron_group_size",
                             "n_genes_per_intron_group"] if c in df_sample_filtered.columns],
        errors="ignore"
    )
    return df_sample_filtered
