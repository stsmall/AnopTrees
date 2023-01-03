# -*-coding:utf-8 -*-
"""
@File    :  test_vcf2tsinfer.py
@Time    :  2022/05/26 10:46:52
@Author  :  Scott T Small
@Version :  1.0
@Contact :  stsmall@gmail.com
@License :  Released under MIT License Copyright (c) 2022 Scott T. Small
@Desc    :  None
@Notes   :  None
@Usage   :  python test_vcf2tsinfer.py
"""
import pandas as pd
import tsinfer
import filecmp
import os
import pytest
import cyvcf2
from AnopTrees.vcf2tsinfer import add_meta_site
from AnopTrees.vcf2tsinfer import add_metadata
from AnopTrees.vcf2tsinfer import add_diploid_sites

# Globals
threads = 1
label_by = "location"

pytest.fixture(autouse=True)
def input_meta():
    return pd.read_csv("tests/meta.test.csv", sep=",", index_col="sample_id", dtype=object)

@pytest.fixture(autouse=True)
def input_vcf():
    return cyvcf2.VCF("tests/tsinfer.test.vcf")

@pytest.fixture(autouse=True)
def input_gff():
    return pd.read_csv("tests/gff.test.csv", sep=",", dtype={'start':int, 'end':int})

def test_add_meta_site(input_gff):
    pos = 100
    meta_pos = add_meta_site(input_gff, pos=pos)
    meta_dt = {'contig': 'X', 'start': 95, 'end': 120, 'type': 'five_prime_UTR', 'name': 'AGAP000002-RA', 'parent': "test", 'description': 'WW domain-containing protein'}
    assert meta_pos == meta_dt

def test_add_diploid_sites(input_vcf, input_gff):
    meta = pd.read_csv("tests/meta.test.csv", sep=",", index_col="sample_id", dtype=object)
    add_diploid_sites(input_vcf, meta, input_gff, 1, "adddiploid_testing", label_by)
    anc = tsinfer.load("adddiploid_testing.samples")
    ts = tsinfer.load("tests/adddiploid.results.samples")
    assert anc.data_equal(ts) == True
    assert(filecmp.cmp("ga.X.exclude-pos.txt", "tests/test.exclude-pos.txt") == True)
    assert(filecmp.cmp("X.missing_data.txt", "tests/test.missing_data.txt") == True)
    assert(filecmp.cmp("X.not_inferred.txt", "tests/test.not_inferred.txt") == True)

    os.remove("adddiploid_testing.samples")
    os.remove("ga.X.exclude-pos.txt")
    os.remove("X.missing_data.txt")
    os.remove("X.not_inferred.txt")