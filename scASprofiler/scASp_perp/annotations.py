import pandas as pd
import numpy as np

def add_gene_annotations(data: pd.DataFrame, gtf_path: str, filter_unique_gene: bool = True) -> pd.DataFrame:
    """
    Add gene_id/gene_name by exon boundary matching (+/-1 trick).
    """
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        header=None,
        comment="#",
        names=[
            "chromosome", "source", "feature",
            "start", "end", "score", "strand", "frame", "attribute"
        ],
    )
    gtf = gtf[gtf.feature == "exon"].copy()
    gtf["gene_id"] = gtf.attribute.str.extract(r'gene_id "([^;]*)";')
    gtf["gene_name"] = gtf.attribute.str.extract(r'gene_name "([^;]*)";')
    gtf["chromosome"] = gtf["chromosome"].astype(str)

    gene_id_name = gtf[["gene_id", "gene_name"]].drop_duplicates()

    exon_starts = gtf[["chromosome", "start", "gene_id"]].copy().rename(columns={"start": "pos"})
    exon_starts["pos"] = exon_starts["pos"] - 1

    exon_ends = gtf[["chromosome", "end", "gene_id"]].copy().rename(columns={"end": "pos"})
    exon_ends["pos"] = exon_ends["pos"] + 1

    exon_boundaries = pd.concat([exon_starts, exon_ends], ignore_index=True).drop_duplicates()
    genes_by_exon_boundary = exon_boundaries.groupby(["chromosome", "pos"]).gene_id.unique()

    data = data.merge(
        genes_by_exon_boundary, how="left",
        left_on=["chromosome", "start"], right_on=["chromosome", "pos"]
    ).rename(columns={"gene_id": "gene_id_start"}).set_index(data.index)

    data = data.merge(
        genes_by_exon_boundary, how="left",
        left_on=["chromosome", "end"], right_on=["chromosome", "pos"]
    ).rename(columns={"gene_id": "gene_id_end"}).set_index(data.index)

    def fill_na_with_empty_array(val):
        return val if isinstance(val, np.ndarray) else np.array([])

    data["gene_id_start"] = data["gene_id_start"].apply(fill_na_with_empty_array)
    data["gene_id_end"] = data["gene_id_end"].apply(fill_na_with_empty_array)

    data["gene_id_list"] = data.apply(
        lambda row: np.unique(np.concatenate([row.gene_id_start, row.gene_id_end])),
        axis=1
    )
    data["n_genes"] = data["gene_id_list"].apply(len)

    data["gene_id_list"] = data["gene_id_list"].apply(lambda x: ",".join(x.tolist()))
    data["gene_id_start"] = data["gene_id_start"].apply(lambda x: ",".join(x.tolist()))
    data["gene_id_end"] = data["gene_id_end"].apply(lambda x: ",".join(x.tolist()))

    if filter_unique_gene:
        print("Filtering to introns associated to 1 and only 1 gene.")
        data = data[data.n_genes == 1].copy()
        data["gene_id"] = data["gene_id_list"]
        data = data.drop(columns=["gene_id_list"])
        data = data.merge(gene_id_name, how="left", on="gene_id").set_index(data.index)
        data.index = data.gene_name.astype(str) + "_" + data.index.astype(str) + data.strand.astype(str)
        data["coordinates"] = (
            data["chromosome"].astype(str) + ":" + data["start"].astype(str) + ":" + data["end"].astype(str)
        )

    return data

