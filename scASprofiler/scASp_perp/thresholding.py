import numpy as np
import pandas as pd
from tqdm import tqdm

META11 = ["chromosome", "strand", "start", "end", "gene_id", "gene_id_start",
          "gene_id_end", "n_genes", "gene_name", "coordinates", "annotated"]


def _cell_block(df: pd.DataFrame) -> pd.DataFrame:

    if "chromosome" not in df.columns:
        raise ValueError(
            "Cannot locate meta start column 'chromosome'. Check your merged dataframe layout.")
    meta_start = df.columns.get_loc("chromosome")

    return df.iloc[:, 1:meta_start].copy()


def thres_filter_group(df_group: pd.DataFrame, samples_ps: int, sites_ps: int) -> pd.DataFrame:
    r = _cell_block(df_group)
    sums = r.sum(axis=0, skipna=True)
    counts = r.count(axis=1)
    r.loc[counts < samples_ps, :] = np.nan
    r.loc[:, sums < sites_ps] = np.nan
    return r


def thres_filter(df: pd.DataFrame, samples_ps: int, sites_ps: int, use_ray: bool = False, num_cpus: int = 50) -> pd.DataFrame:

    groups = df.groupby(by="intron_group", sort=False)

    if use_ray:
        import ray
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

        @ray.remote
        def worker(gdf, samples_ps, sites_ps):
            return thres_filter_group(gdf, samples_ps, sites_ps)

        futures = [worker.remote(s[1], samples_ps, sites_ps)
                   for s in tqdm(groups)]
        mats = ray.get(futures)
        ray.shutdown()
    else:
        mats = [thres_filter_group(s[1], samples_ps, sites_ps)
                for s in tqdm(groups)]

    result = pd.concat(mats, axis=0)

    result.insert(0, "coord.intron", df.loc[result.index, "index"].values)

    for c in META11:
        if c in df.columns:
            result.loc[result.index, c] = df.loc[result.index, c].values

    return result
