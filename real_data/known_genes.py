# ============================================================
# real_data/known_genes.py
#
# Known cancer genes used for interpretation
# ============================================================

# Lung adenocarcinoma genes
LUAD_GENES = [
    "EGFR",
    "KRAS",
    "TP53",
    "ALK",
    "BRAF",
    "MET",
    "ERBB2",
    "PIK3CA",
    "STK11",
]

# Breast cancer genes
BRCA_GENES = [
    "BRCA1",
    "BRCA2",
    "TP53",
    "PIK3CA",
    "PTEN",
    "ERBB2",
    "AKT1",
    "GATA3",
    "MAP3K1",
]


def get_known_genes(dataset_name):

    if dataset_name.lower() == "luad":
        return LUAD_GENES

    if dataset_name.lower() == "brca":
        return BRCA_GENES

    return []


def filter_known_genes_present(gene_list, dataset_genes):
    """
    Return only known genes that are present in the dataset.
    """

    dataset_genes = set(dataset_genes)

    return [g for g in gene_list if g in dataset_genes]
