inv_dt = {"2La":(20_524_058, 42_165_532), "2Rb":(19_023_925, 26_758_676), 
        "2Rc":(26_750_000, 31_473_000), "2Rd":(31_480_500, 42_600_500), 
        "2Rj":(3_262_186, 15_750_716), "2Ru":(31_473_000, 35_505_236)}
offset = 1_000_000
chrom = "2L"
chrom_start = inv_dt["2La"][0] - offset
chrom_end = inv_dt["2La"][1] + offset
win_size = 500_000
win_step = 100_000
group = "country"
country = "Democratic Republic of the Congo"
color_by = "2La"
meta_path = '/home/ssmall/projects/anop_trees/ag1000g_p3/An_gambiae.meta.ck.csv'
zarr_path = '/home/ssmall/projects/anop_trees/ag1000g_p3/AgamP3.phased.zarr'
phased = True
outfile = "2La.DRC-phased"