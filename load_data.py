inpath = '../data/ufo_awesome.tsv'
outpath = '../data/ufo_awesome_6col.tsv'

def ufotabs_to_sixcols(inpath, outpath):
	inf = open(inpath, 'r')
	outf = open(outpath, 'w')
	
	for line in inf:
		splitline = line.split('\t')
		if len(splitline) < 6:
			continue
		newline = '\t'.join(splitline[:6])
		if newline[-1:] != '\n':
			newline += '\n'
		outf.write(newline)
	inf.close()
	outf.close()

ufotabs_to_sixcols(inpath, outpath)