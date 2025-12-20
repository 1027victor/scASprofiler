import pandas as pd
# import numpy as np
from .constants import STAR_SJ_COLUMNS

def process_splicing_junctions(sj_path: str) -> pd.DataFrame:
    """
    Load and preprocess STAR SJ.out.tab-like file (9 columns).
    """
    sj = pd.read_csv(sj_path, header=None, sep="\t")
    sj.columns = STAR_SJ_COLUMNS

    sj["strand"] = sj["strand"].replace({0: "NA", 1: "+", 2: "-"})
    sj.index = sj["chromosome"].astype(str) + ":" + sj["start"].astype(str) + "-" + sj["end"].astype(str)

    sj = sj[sj["strand"] != "NA"]
    sj = sj.drop_duplicates(["chromosome", "start", "end", "strand"])
    sj = sj[sj["chromosome"].astype(str).str.startswith("chr") & (sj["chromosome"] != "chrM")]
    sj = sj.sort_values(by=["chromosome", "start", "end"])
    return sj

