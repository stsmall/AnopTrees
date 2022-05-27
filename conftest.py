import pytest
import cyvcf2
import pandas as pd
from AnopTrees.vcf2tsinfer import add_meta_site
from AnopTrees.vcf2tsinfer import add_metadata
from AnopTrees.vcf2tsinfer import add_diploid_sites

pytest.fixture(autouse=True)
def input_meta():
    return pd.read_csv("tests/meta.test.csv", sep=",", index_col="sample_id", dtype=object)

@pytest.fixture(autouse=True)
def input_vcf():
    return cyvcf2.VCF("tests/tsinfer.test.vcf")

@pytest.fixture(autouse=True)
def input_gff():
    return pd.read_csv("tests/gff.test.csv", sep=",", dtype={'start':int, 'end':int})
