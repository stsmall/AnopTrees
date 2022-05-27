
import sys
chrom = sys.argv[1] 
stop = int(sys.argv[2])
gff = sys.argv[3]


def load_gff(chrom, stop, gff, gff_filt):
    """Open gff file and keep coordinate list of features."""
    feat_ls = ["gene", "exon", "CDS", "mRNA", "five_prime_UTR", "three_prime_UTR"]
    if gff_filt is None:
        gff_filt = feat_ls
    gff_ls = []
    e = 1
    intron_s = 1
    gs = 1
    with open(gff, 'r') as gf:
        for line in gf:
            if not line.startswith("#"):
                g_lin = line.strip().split(",")
                if g_lin[0] == chrom:
                    feat = g_lin[2]
                    if "intergenic" in gff_filt:
                        if feat == "gene":
                            if gs == 1:
                                gff_ls.append((gs, int(g_lin[3]), "intergenic",'','',''))
                                gs = int(g_lin[4])
                            else:
                                if gs < int(g_lin[3]):
                                    gff_ls.append((gs, int(g_lin[3]), "intergenic",'','',''))
                                    gs = int(g_lin[4])
                    elif feat in feat_ls:
                        s = int(g_lin[3])
                        e = int(g_lin[4])
                        if feat in gff_filt:
                            gff_ls.append((s, e, feat, g_lin[8], g_lin[9], g_lin[10]))
                        if "intron" in gff_filt:
                            if "exon" in feat:
                                if intron_s == 1:
                                    intron_s = e
                                else:
                                    gff_ls.append((intron_s, s, "intron",'','',''))
                                    intron_s = e
    if "intergenic" in gff_filt and e < stop:
        gff_ls.append((gs, stop, "intergenic",'','',''))

    return gff_ls

f = open("gff_reformat", 'w')
f.close()

for filt in ["intron", "intergenic", None]:
    gff_ls = load_gff(chrom, stop, gff, filt)
    with open("gff_reformat", 'a') as f:
        for s, e, feat, par, name, desc in gff_ls:
            f.write(f"{chrom},{s},{e},{feat},{par},{name},{desc}\n")