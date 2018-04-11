#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from ufo_reflection import *

if __name__ == '__main__':
	sources=load_sources('data.tsv')

	print '\nName\t&\t $v_\mathrm{UFO}$ (c)\t&\tReference \t&\t $i$ (degrees) \t&\t Reference\\\\'
	print '\hline'
	for sname in sorted(sources.iterkeys()):
		source=sources[sname]

		source.print_latex()