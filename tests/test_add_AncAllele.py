import pytest
import filecmp
import os

from AnopTrees.add_AncAllele import read_estsfs
from AnopTrees.add_AncAllele import add_aa

@pytest.fixture()
def est_infile():
    return read_estsfs("tests/test.aa.txt")
    
def test_add_aa(est_infile):
    add_aa(est_infile, "tests/test.aa.vcf.gz")
    assert(filecmp.cmp("tests/result.aa.derived.vcf", "tests/test.aa.derived.vcf"))
    os.remove("tests/test.aa.derived.vcf")