CHROMS = ["chr2", "chr3", "chrX"]

rule all:
	expand("KirFol.AfunF3.{chroms}.pop.filter.mask.dp10gq20.keep.kf170.p10.phased.anc.vcf.gz", chroms=CHROMS)
	expand("KirFol.AfunF3.{chroms}.pop.filter.mask.dp10gq20.keep.kf170.p10.anc.vcf.gz", chroms=CHROMS)

rule count_allele:
	message: "counting outgroup alleles for polarizing"
	input:
		outgroup_vcf_1 = "Anriv.{chroms}.vcf.gz",
		outgroup_vcf_2 = "Aspa.{chroms}.vcf.gz",
		vcf = "KirFol.AfunF3.{chroms}.pop.filter.mask.dp10gq20.keep.kf170.p10.vcf.gz"
	output:
	        "Ariv.{chroms}.allele.counts.txt.gz",
		"Aspa.{chroms}.allele.counts.txt.gz",
		"KirFol.AfunF3.{chroms}.allele.counts.txt.gz"
	shell:
		"""
		vcftools --gzvcf {input.outgroup_vcf_1} --counts --stdout | gzip -c > Ariv.{chroms}.allele.counts.txt.gz
		vcftools --gzvcf {input.outgroup_vcf_2} --counts --stdout | gzip -c > Aspa.{chroms}.allele.counts.txt.gz
		vcftools --gzvcf {input.vcf} --counts --stdout | gzip -c > KirFol.AfunF3.{chroms}.allele.counts.txt.gz
		"""

rule run_makeinfile:
	input:
		riv = "Ariv.{chroms}.allele.counts.txt.gz",
 		spa = "Aspa.{chroms}.allele.counts.txt.gz",
		fun = "KirFol.AfunF3.{chroms}.allele.counts.txt.gz"
	output:
		"{wildcards.chroms}.estsfs.data.txt"
	shell:	
		"""
		python estsfs_input.py --ingroup {input.fun} --outgroups {input.riv} {input.spa}
		"""

rule run_estsfs:
	input:
		rules.run_makeinfile.output
	output:
		"{wildcards.chroms}.estsfs.out"
	shell:
		"""
		module load parallel
		split -l 100000 {wildcards.chroms}.est.infile {wildcards.chroms}-estsfs
		ls -l {wildcards.chroms}-estsfs* | parallel -P 40 -N1 '~/programs_that_work/est-sfs-release-2.03/est-sfs config-file.txt {} seedfile.txt {wildcards.chroms}.{}.sfs {wildcards.chroms}.{}.pvalues'
		for p in *.pvalues;do
		    grep -v "^0" $p >> {wildcards.chroms}.estsfs.pvalues
		done
		paste <( paste < {wildcards.chroms}.pos.txt ) <( cut -d" " -f3- {wildcards.chroms}.estsfs.pvalues )  > {wildcards.chroms}.estsfs.out.txt
		"""

rule add_derived:
	input:
		rules.run_estsfs.output,
		vcf = "KirFol.AfunF3.{chroms}.pop.filter.mask.dp10gq20.keep.kf170.p10.vcf.gz"
	output:
		"KirFol.AfunF3.{chroms}.pop.filter.mask.dp10gq20.keep.kf170.p10.anc.vcf.gz"
	shell:
		"""
		python ~/programs_that_work/KirFol/derivedVCF.py --vcfFile {input.vcf} --estFile {wildcards.chroms}.estsfs.out.txt
		"""



