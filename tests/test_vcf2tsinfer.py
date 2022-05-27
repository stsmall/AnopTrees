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
import pytest
import cyvcf2
import pandas as pd
import tsinfer 

from vcf2tsinfer import add_meta_site
from vcf2tsinfer import add_metadata
from vcf2tsinfer import add_diploid_sites

    # sample.data_equal()
    # sample.info()
    # sample.sites()
    # sample.num_alleles
    # sample.haplotypes

# Globals
threads = 1
label_by = "location"

pytest.fixture()
def input_meta():
    meta = pd.read_csv("gamb.test.meta.csv", sep=",", index_col="sample_id", dtype=object)

@pytest.fixture()   
def input_vcf():
    vcf = cyvcf2.VCF("test.aa.vcf.gz")

@pytest.fixture()   
def input_gff():
    gff = pd.read_csv("gamb.test.gff", sep=",", dtype={'start':int, 'end':int})

def test_addmetadata(input_vcf, input_meta):
    vcf = input_vcf
    meta = input_meta
    sample_data = tsinfer.SampleData("addmeta.test")
    sample_data = add_metadata(vcf, sample_data, meta, label_by)
    sample_data.finalise()
    anc = tsinfer.load("adddiploid.test.samples")
    #assert sample_data.() samples have correct data from meta
    
def test_add_meta_site(input_gff):
    pos = 10000
    meta_pos = add_meta_site(input_gff, pos=pos)
    meta_dt = {}
    assert meta_pos == meta_dt
    
def test_add_diploid_sites(input_vcf, input_meta, input_gff):
    add_diploid_sites(input_vcf, input_meta, input_gff, 1, "adddiploid.test", label_by)
    ts = tsinfer.load("adddiploid.test.samples")
    # test sites
    for s in ts.sites():
        pass
    
    
    assert anc.data_equal(tsinfer.load("test.samples")) == True
