#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
import warnings
from math import log10, floor
from bayesian_regression import *
from scipy.optimize import root


def cone(v,i,iw=30):
	output=v*np.cos(i/180.*np.pi-iw/180.*np.pi)
	output[i<iw]=0
	return output

def slab(v,i,iw=70):
	output=v*np.cos(-i/180.*np.pi+iw/180.*np.pi)
	output[i>iw]=0
	return output

def round_sigfigs(num, sig_figs):
    if num != 0:
        return round(num, -int(floor(log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0


class ufo_record():
	"""Class for holding UFO data from data file, needs a velocity, error and reference"""
	def __init__(self, velocity, error=0.01, reference='None',merged=False):
		# super(ufo_record, self).__init__()
		self.velocity = velocity
		self.merged=merged
		if hasattr(error, '__iter__'):
			if len(error)>2 or len(error)==0:
				raise ValueError('error must be a single value or a length 2 iterable (+err,-err)')
			else:
				self.error=(float(error[0]),float(error[1]))
		else:
			self.error=(float(error),float(error))
		self.reference=str(reference)
	def latex_print(self):
		v_str='$'+str(round_sigfigs(self.velocity,3))
		if self.error[0]==self.error[1]:
			err_str='\pm'+str(round_sigfigs(self.error[0],2))
		else:
			err_str='^{+'+str(round_sigfigs(self.error[0],2))+'}_{-'+str(round_sigfigs(self.error[1],2))+'}'
		ref_str='; '.join([x[0]+x.split('+')[-1] for x in self.reference.split('; ')])
		# ref_str='\citet{'+'}; \citet{'.join([x.split('+')[0]+x.split('+')[-1] for x in self.reference.split('; ')])+'}'
		out_str=v_str+err_str+'$\t&\t'+ref_str
		return out_str

		
class reflection_record():
	"""As above, for reflection inclinations"""
	def __init__(self, inclination, error=5,reference='None',qc=True):
		# super(reflection_record, self).__init__()
		self.inclination=inclination
		self.reference=reference
		self.qc=qc # Does source meet Reynolds quality control?
		if hasattr(error, '__iter__'):
			if len(error)>2 or len(error)==0:
				raise ValueError('error must be a single value or a length 2 iterable (+err,-err)')
			else:
				self.error=(float(error[0]),float(error[1]))
		else:
			self.error=(float(error),float(error))

	def latex_print(self):
		i_str='$'+str(round_sigfigs(self.inclination,3))
		if self.error[0]==self.error[1]:
			err_str='\pm'+str(round_sigfigs(self.error[0],2))
		else:
			err_str='^{+'+str(round_sigfigs(self.error[0],2))+'}_{-'+str(round_sigfigs(self.error[1],2))+'}'
		ref_str='; '.join([x[0]+x.split('+')[-1] for x in self.reference.split('; ')])
		# ref_str='\citet{'+'}; \citet{'.join([x.split('+')[0]+x.split('+')[-1] for x in self.reference.split('; ')])+'}'
		out_str=i_str+err_str+'$\t&\t'+ref_str
		return out_str

	
def weighted_mean(x,errors):
	"""Calculate the weighted mean and uncertainty from multiple measurements x with errors e"""
	# Note that asymmetric errors are averaged, which is technically not correct.
	# A generally applicable solution to this does not exist, as error formulae generally
	# assume a normal distribution. As far as I am aware, there is no 'correct' way of combining 
	# these errors without knowing the posterior distribution

	# If positive and negative errors are given separately, average
	if len(errors.shape)==2:
		errors=np.mean(errors,axis=1)

	weights=[1./e**2 for e in errors]

	w_mean=sum([xi*wi for xi,wi in zip(x, weights)])/sum(weights)
	w_err=1./(sum(weights))**0.5

	return w_mean, w_err

class Source():
	"""Class for holding info on a given AGN"""
	def __init__(self, name, ufo_velocity_threshold=0.033):
		# super(Source, self).__init__()
		self.name = name
		self.ufos=[]
		# self.inclination=[]
		self.ufo_velocity_threshold=ufo_velocity_threshold
		self.n_ufos=0
		self.has_reflection=False

	def add_ufo(self,velocity,error,reference):
		if velocity>self.ufo_velocity_threshold:
			self.ufos.append(ufo_record(velocity,error,reference))
			self.n_ufos+=1
		else:
			self.velocity_warning(velocity)

	def add_refl(self,inclination,error,reference,qc):
		self.has_reflection=True
		self.reflection=reflection_record(inclination,error,reference,qc)


	def velocity_warning(self,v):
		# warnings.warn('WARNING: Velocity below threshold, record not added')
		print 'Skipped:',self.name,'v=',v,'threshold=',self.ufo_velocity_threshold

	def consolidate_ufos(self,merge_velocity=0.01,merge_percent=10.0):
		"""Merges any UFOs that lie within merge_velocity, merge_percent, or error"""
		if self.n_ufos>1:
			velocities=[x.velocity for x in self.ufos]
			errors=[x.error for x in self.ufos]
			errors=[y for x,y in sorted(zip(velocities,errors))]
			sorted_ufos=[y for x, y in sorted(zip(velocities,self.ufos))]
			velocities=sorted(velocities)
			merge_flags=[]

			# Determine which UFOs to merge. 
			for i in range(0,len(velocities)-1):
				v1=velocities[i]
				v2=velocities[i+1]
				e1=errors[i]
				e2=errors[i+1]
				# Merge those within merge_percent 
				if v1+v1*merge_percent/100.>=v2 or v2-v2*merge_percent/100.<=v1:
					merge_flags.append(1)
				# Merge those within merge velocity
				elif v1+merge_velocity>=v2:
					merge_flags.append(1)
				# Merge those within error
				elif v1+e1[0]>=v2-e2[1]:
					merge_flags.append(1)
				else:
					merge_flags.append(0)
			merge_flags.append(0)

			if sum(merge_flags)==0:
				pass # No merges happen

			else:
				print 'Merging UFOs...'
				new_ufos=[]

				temp_vs=[]
				temp_refs=[]
				temp_errors=[]

				start_merge=False
				for i,v in enumerate(velocities):
					mf=merge_flags[i]
					if start_merge==False:
						if mf==0:
							new_ufos.append(sorted_ufos[i])
						else:
							start_merge=True
							temp_vs.append(v)
							temp_refs.append(sorted_ufos[i].reference)
							temp_errors.append(sorted_ufos[i].error)
					elif start_merge==True:
						temp_vs.append(v)
						temp_refs.append(sorted_ufos[i].reference)
						temp_errors.append(sorted_ufos[i].error)
						if mf==0:
							start_merge=False

							# new_v=np.mean(temp_vs)
							new_ref='; '.join(list(set(temp_refs)))
							# err=np.std(temp_vs)
							# print self.name
							w_mean, w_err=weighted_mean(temp_vs,np.array(temp_errors))
							# exit()

							new_ufos.append(ufo_record(w_mean,w_err,new_ref,merged=True))

							temp_vs=[]
							temp_errors=[]
							temp_refs=[]

				self.ufos=new_ufos
				self.n_ufos=len(new_ufos)
				print 'Consolodated to',self.n_ufos,'UFOs'

		else:
			pass

	def display_ufos(self):
		for ufo in self.ufos:
			if ufo.error[0]==ufo.error[1]:
				print str(round_sigfigs(ufo.velocity,3))+'+/-'+str(round_sigfigs(ufo.error[0],1)), ufo.reference
			else:
				print str(round_sigfigs(ufo.velocity,3))+'+'+str(round_sigfigs(ufo.error[0],1))+'-'+str(round_sigfigs(ufo.error[1],1)), ufo.reference

	def display_refl(self):
		if self.has_reflection:
			refl=self.reflection
			if refl.error[0]==refl.error[1]:
				print str(refl.inclination)+'+/-'+str(refl.error[0]), refl.reference
			elif refl.inclination+refl.error[0]>=90:
				print '>'+str(refl.inclination-refl.error[1]), refl.reference
			elif refl.inclination-refl.error[1]<=0:
				print '<'+str(refl.inclination+refl.error[0]), refl.reference
			else:
				print str(refl.inclination)+'+'+str(refl.error[0])+'-'+str(refl.error[1]), refl.reference
		else:
			print 'Reflection not loaded'

	def print_latex(self):
		# print self.n_ufos
		for i,ufo in enumerate(self.ufos):
			if i==0:
				print self.name+'\t&\t'+ufo.latex_print()+'\t&\t'+self.reflection.latex_print()+'\\\\'
			else:
				print '\t&\t'+ufo.latex_print()+'\\\\'




def load_data(filename,delimiter='\t',skiprows=2):
	indata=np.loadtxt(filename,delimiter=delimiter,skiprows=skiprows,dtype='str')
	return indata

def get_sourcenames(data):
	return sorted(list(set(data[:,0])))

def get_source_data(data,source_name):
	return data[data[:,0]==source_name]

def parse_ufo_data(record):
	v=float(record[3])
	pos_err=float(record[4])
	neg_err=float(record[5])
	ref=record[7]
	return v,(pos_err,neg_err),ref

def parse_refl_data(record):
	i=float(record[9])
	pos_err=float(record[10])
	neg_err=float(record[11])
	ref=record[12]
	qc=record[13]
	if qc in ['y','Y']:
		qc=True
	if qc in ['n','N']:
		qc=False
	return i,(pos_err,neg_err),ref,qc

def get_points(source_dict,i_syserror=5.):
	"""Function to get points and limits from dictionary of source names and source objects"""
	print '\nGetting points and limits'
	vs=[]
	v_errs=[]
	points=[]
	errs=[]

	ulim_vs=[]
	ulim_v_errs=[]
	ulims=[]

	llim_vs=[]
	llim_v_errs=[]
	llims=[]

	for sname in source_dict:
		source=source_dict[sname]
		n_ufos=source.n_ufos
		if n_ufos>0:
			
			print sname, n_ufos, 'UFOs'
			i=source.reflection.inclination
			i_err=source.reflection.error
			if i_err[0]<i_syserror:
				i_err=(i_syserror,i_err[1])
			
			if i_err[1]<i_syserror:
				i_err=(i_err[0],i_syserror)


			for ufo in source.ufos:
				v=ufo.velocity
				v_err=ufo.error

				if i+i_err[0]>=90.:
					llims.append(i-i_err[1])
					llim_vs.append(v)
					llim_v_errs.append(v_err)
					
				elif i-i_err[1]<=0.:
					ulims.append(i+i_err[0])
					ulim_vs.append(v)
					ulim_v_errs.append(v_err)

				else:
					points.append(i)
					errs.append(i_err)
					vs.append(v)
					v_errs.append(v_err)
		else:
			pass

	return vs, np.array(v_errs), points, np.array(errs), llim_vs, np.array(llim_v_errs), llims, ulim_vs, np.array(ulim_v_errs), ulims


def plot_relation(sources,filename='UFO_relation.pdf'):
	vs, v_errs, points, errs, llim_vs, llim_v_errs, llims, ulim_vs, ulim_v_errs, ulims = get_points(sources)
	print '\nPlotting'
	a,b,aerr,berr,new_sample=likelyhood_calc(vs,points, errs,xerrors=v_errs,upperlimits_x=ulim_vs, upperlimits_y=ulims,alims=(-100,200),blims=(0,70),n_samples=1000)


	# Plot normal points:
	pl.clf()
	fig=pl.figure(figsize=(6,5))
	ax1=pl.subplot(111)
	ax1.set_ylim(0,0.5)
	ax1.set_xlim(0,75)
	ax1.set_xlabel('Inclination (degrees)')
	# ax1.set_xscale('log')
	ax1.set_ylabel('UFO velocity (c)')
	pl.errorbar(points,vs,v_errs.T,errs.T,ls='none',lw=1,color='k')

	# Plot upper limits:
	if len(ulims)>0:
		pl.errorbar(ulims,ulim_vs,ulim_v_errs,0,ls='none',lw=1,color='k')

		pl.errorbar(ulims, ulim_vs,0, 5, xuplims=True,ls='none',lw=1,color='k')

	
	# print a,b, aerr, berr
	# print new_sample
	# print [0.,b],[0.5,0.5*a+b]
	ax1.plot([b,0.5*a+b],[0.,0.5],color='dodgerblue')
	xs=np.linspace(0,0.5,201)
	max_is=[]
	min_is=[]
	std_devs=[]
	means=[]
	for x in xs:
		ys=new_sample[:,0]*x+new_sample[:,1]
		max_is.append(max(ys))
		min_is.append(min(ys))
		std_devs.append(np.std(ys))
		means.append(np.mean(ys))
	std_devs=np.array(std_devs)
	means=np.array(means)
	pl.fill_betweenx(xs,means+std_devs,means-std_devs,color='dodgerblue',alpha=0.3)

	# inclinations=np.linspace(0,90,180)

	# pl.show()
	# pl.savefig(filename,bbox_inches='tight')
	# pl.show()
	# print 'Saved to',filename	

def load_sources(data_file):
	ufo_data=load_data('data.tsv')
	source_names=get_sourcenames(ufo_data)
	
	# Define a Source object for each source name
	sources={}
	for sname in source_names:
		sources[sname]=Source(sname)

	# Add data to each source
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
	return sources

def main():
	# Load data from file and get list of source names:
	sources=load_sources('data.tsv')

	# Get points and upper limits for all sources
	plot_relation(sources)
	pl.savefig('test.pdf',bbox_inches='tight')

	pass

if __name__ == '__main__':
	main()