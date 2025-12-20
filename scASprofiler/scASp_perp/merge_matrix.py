import os
import glob
import pandas as pd
# import numpy as np
# from typing import List, Tuple
from typing import List
from .sj_processing import process_splicing_junctions

def discover_sj_files(sj_dir: str) -> List[str]:
    patterns = [
        os.path.join(sj_dir, "*.tab"),
        # os.path.join(sj_dir, "*.sj.out.tab"),
        # os.path.join(sj_dir, "**", "*.SJ.out.tab"),
        # os.path.join(sj_dir, "**", "*.sj.out.tab"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set([f for f in files if os.path.isfile(f)]))
    if not files:
        raise FileNotFoundError(f"No *.SJ.out.tab files found under: {sj_dir}")
    return files


def build_row_merged_out_tab(sj_files: List[str], out_path: str) -> str:
    dfs = []
    for f in sj_files:
        df = process_splicing_junctions(f).reset_index(drop=False).rename(columns={"index": "sj_id"})
        df["source_file"] = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    merged = merged.drop_duplicates(["chromosome", "start", "end", "strand"])
    merged = merged.sort_values(["chromosome", "start", "end"])
    merged = merged[["chromosome","start","end","strand","intron_motif","annotated","total_unique_mapping","total_multi_mapping","max_overhang"]]
    # merged.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path

def build_junction_by_sample_matrix(
    sj_files: List[str],
    out_path: str,
    use_multi: bool = False,
) -> str:
    series_dict = {}
    for f in sj_files:
        base = os.path.basename(f)
        name = base.replace(".SJ.out.tab", "").replace(".sj.out.tab", "")

        sj = process_splicing_junctions(f).copy()
        sj["coordinates"] = sj["chromosome"].astype(str) + ":" + sj["start"].astype(str) + ":" + sj["end"].astype(str)

        if use_multi:
            v = (sj["total_unique_mapping"] + sj["total_multi_mapping"]).astype(float).values
        else:
            v = sj["total_unique_mapping"].astype(float).values

        s = pd.Series(v, index=sj["coordinates"].values, name=name)
        s = s.groupby(level=0).sum()
        series_dict[name] = s

    mat = pd.DataFrame(series_dict)
    mat.index.name = "coordinates"
    mat.to_csv(out_path, sep="\t")
    return out_path

