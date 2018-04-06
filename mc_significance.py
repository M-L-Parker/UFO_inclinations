#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from ufo_reflection import *
from scipy.stats import pearsonr, spearmanr


def run_mc(sources,N):
	print '\nRunning MC significance analysis'
	vs, v_errs, points, errs, llim_vs, llim_v_errs, llims, ulim_vs, ulim_v_errs, ulims = get_points(sources)
	# This will ignore lower limits, as we only have upper limits
	# Upper limits are treated as values of zero, with errors of 10.
	for i, ulim in enumerate(ulims):
		np.append(points, 0.)
		np.append(errs, ulim)
		np.append(vs, ulim_vs[i])
		np.append(ulim_v_errs, ulim_v_errs[i])

	# Calculate Pearson and Spearman correlation coefficients for real data:
	p=pearsonr(vs,points)
	s=spearmanr(vs,points)

	print '\nPearson r:', p[0]
	print 'P-value:',p[1]

	print '\nSpearman r:', s[0]
	print 'P-value:',s[1]

	# Using log(v), gives a more normal distribution...
	# ax1=pl.subplot(121)
	# pl.hist(np.log10(vs))
	logv_mean=np.mean(np.log10(vs))
	logv_std=np.std(np.log10(vs))
	print '\nMean log(v):',logv_mean
	print 'Standard deviation:',logv_std

	# ax2=pl.subplot(122)
	# pl.hist(points)
	i_mean=np.mean(points)
	i_std=np.std(points)
	print '\nMean i:',i_mean
	print 'Standard deviation:',i_std

	pl.savefig('parameter_distributions.pdf',bbox_inches='tight')

	# Simulate N sets of points, randomly selected from Gaussian distributions:
	fake_vs=10.**np.random.normal(logv_mean,logv_std,size=(len(vs),N))
	fake_is=np.random.normal(i_mean,i_std,size=(len(vs),N))	
	# print fake_vs

	pearsonr_sims=[]
	spearmanr_sims=[]
	for i in range(0,N):
		pearsonr_sims.append(pearsonr(fake_vs[:,i],fake_is[:,i])[0])
		spearmanr_sims.append(spearmanr(fake_vs[:,i],fake_is[:,i])[0])
	
	print '\nResults:'
	N_p=len([x for x in pearsonr_sims if abs(x)>p[0]])
	N_s=len([x for x in spearmanr_sims if abs(x)>s[0]])
	print N_p,'of %s simulations exceeded Pearson r value of %s' %(str(N),str(p[0]))
	print 'P =',round_sigfigs(float(N_p)/float(N),3)
	print N_s,'of %s simulations exceeded Spearman r value of %s' %(str(N),str(s[0]))
	print 'P =',round_sigfigs(float(N_s)/float(N),3)

	fig2=pl.figure(figsize=(8,5))
	ax1=pl.subplot(121)
	ax1.set_label('N')
	ax1.set_xlabel('Pearson r')
	ax1.set_xlim(0,1)
	pl.hist([abs(x) for x in pearsonr_sims],bins=100)
	ax2=pl.subplot(122)
	ax2.set_xlim(0,1)
	ax2.set_label('N')
	ax2.set_xlabel('Spearman r')
	pl.hist([abs(x) for x in spearmanr_sims],bins=100)
	# pl.savefig('sim_coeff_distributions.pdf',bbox_inches='tight')

	pl.show()






# Load data from file and get list of source names:
# ufo_data=load_data('data.tsv')
if __name__ == '__main__':

	ufo_data=load_data('data.tsv')
	source_names=get_sourcenames(ufo_data)

	# Define a Source object for each source name
	sources={}
	for sname in source_names:
		sources[sname]=Source(sname)

	for sname in sources:

		print '\nAnalysing',sname
		source=sources[sname]
		source_data=get_source_data(ufo_data,sname)
		for record in source_data:
			velocity_temp,err_temp,ref_temp=parse_ufo_data(record)
			source.add_ufo(velocity_temp,err_temp,ref_temp)
			if not source.has_reflection:
				i_temp,err_temp,ref_temp,qc=parse_refl_data(record)
				source.add_refl(i_temp,err_temp,ref_temp,qc)
				print 'Reflection loaded'
				source.display_refl()

		print source.n_ufos,'UFOs loaded'
		source.consolidate_ufos()
		source.display_ufos()

	run_mc(sources,50000)