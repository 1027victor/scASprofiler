import numpy as np
import pandas as pd

def _to_int_str(x: pd.Series) -> pd.Series:
    xi = pd.to_numeric(x, errors="coerce").round().astype("Int64")
    return xi.astype(str)

def _first_col(df: pd.DataFrame, col: str) -> pd.Series:
    x = df[col]
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def group_introns(data_intron, by="three_prime", filter_unique_gene_per_group=True):
    data_intron = data_intron.copy()  
    if data_intron.columns.duplicated().any():
        dup = data_intron.columns[data_intron.columns.duplicated()].tolist()
        print(f"[WARN] duplicated columns detected in group_introns(): {dup} (will use first occurrence)")
    chrom = _first_col(data_intron, "chromosome").astype(str)
    start = _to_int_str(_first_col(data_intron, "start").astype(str))
    end = _to_int_str(_first_col(data_intron, "end").astype(str))
    strand = _first_col(data_intron, "strand").astype(str)

    if by == "three_prime":
        pos = np.where(strand.values == "+", end.values, start.values)
        data_intron["intron_group"] = chrom.values + "_" + pos.astype(str) + "_" + strand.values

    elif by == "five_prime":
        pos = np.where(strand.values == "-", end.values, start.values)
        data_intron["intron_group"] = chrom.values + "_" + pos.astype(str) + "_" + strand.values

    elif by == "gene":
        gid = _first_col(data_intron, "gene_id").astype(str)
        data_intron["intron_group"] = gid.values

    else:
        raise Exception(f"Grouping by {by} not yet supported.")

    intron_group_sizes = (
        data_intron.intron_group.value_counts()
        .rename("intron_group_size")
        .to_frame()
    )
    data_intron = data_intron.merge(
        intron_group_sizes, how="left", left_on="intron_group", right_index=True
    ).set_index(data_intron.index)

    print("Filtering singletons.")
    data_intron = data_intron[data_intron.intron_group_size > 1]

    if filter_unique_gene_per_group:
        print("Filtering intron groups associated with more than 1 gene.")
        n_genes_per_intron_group = (
            data_intron.groupby("intron_group").gene_id.nunique().to_frame()
            .rename(columns={"gene_id": "n_genes_per_intron_group"})
        )
        data_intron = data_intron.merge(
            n_genes_per_intron_group, how="left", left_on="intron_group", right_index=True
        )
        data_intron = data_intron[data_intron.n_genes_per_intron_group == 1]
        data_intron.intron_group = data_intron.gene_name.astype(str) + "_" + data_intron.intron_group.astype(str)

    return data_intron
