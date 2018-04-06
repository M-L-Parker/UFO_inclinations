#!/usr/bin/env python
import numpy as n
from matplotlib import pyplot as p
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d

def gauss(x,a,x0,sigma):
    return a*n.exp(-(x-x0)**2/(2*sigma**2))


def draw_samples(x,N):
	x=x/x.max()
	# N_found=0
	# print x.max()
	found_values=[]
	while len(found_values)<N:
		i,j=n.random.randint(0,x.shape[0]),n.random.randint(0,x.shape[1])
		prob=x[i,j]
		if n.random.rand()<prob:
			found_values.append((i,j))
	return n.array(found_values)


def line_distance(x0,y0,m,c):
	A=-m
	B=1.
	C=-c
	d=abs(A*x0+B*y0+C)/(A**2+B**2)**0.5
	return d
    
        
# see http://www.astroml.org/book_figures/chapter8/fig_linreg_inline.html for general principle
# the exact code there seems clunky, because the example only uses 4 points.

def likelyhood_calc(xpoints,ypoints,yerrors,xerrors=None,upperlimits_x=None,upperlimits_y=None, alims=(0,1),blims=(0,1),size=100,stem=None,n_samples=1000):
	"""bayesian regression. WIP, needs limits implementing on other pars"""
	### Still very rough. Will generally only work if carefully tuned to the parameter space. I'll improve it one day.
	### 

	if stem is None:
		stem='probability_dist'
	a_range=n.linspace(alims[0],alims[1],size)
	b_range=n.linspace(blims[0],blims[1],size) ## for y=ax+b
	
	# new_errors=n.zeros(yerrors.shape)



	fig_regression=p.figure('Parameter space',figsize=(8,8))
	p.clf()
	prob_array=n.ones((size,size))

	ax1=p.subplot(223)
	ax1.set_ylim(blims)
	ax1.set_xlim(alims)
	ax1.set_xlabel('a')
	ax1.set_ylabel('b')

	# Constraints from points
	for i,ypoint in enumerate(ypoints):
		# print ypoint,yerrors[:,i].mean()
		xpoint=xpoints[i]
		# print yerrors
		yerr=yerrors[i].mean()
		if xerrors is not None:
			xerr=xerrors[i].mean()

		# d_sigma=yerr/(1.+xpoint**2)**0.5

		temp_array=n.zeros((size,size))

		for l in range(0,size):
			a=a_range[l]
			for m in range(0,size):
				b=b_range[m]
				

				if xerrors is not None:
					# likelihood is the gaussian PDF of the distance from the line a=-xb+y of the point a0,b0, divided by the error in the distance
					d=line_distance(a,b,-xpoint,ypoint)
					d_err=((line_distance(a,b,-(xpoint+xerr),ypoint)-d)**2+(line_distance(a,b,-xpoint,ypoint+yerr)-d)**2)**0.5
				else:

					d=abs(b+xpoint*a-ypoint)
					d_err=yerr

				# print d, d_err

				temp_array[l,m]=norm.pdf(d/d_err)

		prob_array=temp_array*prob_array
		# p.imshow(prob_array)

		p.plot(a_range,a_range*-xpoint+ypoint,color='k',alpha=0.3)

	# Constraints from  limits:

	if upperlimits_y is not None:
		for i, ylim in enumerate(upperlimits_y):
			xpoint=upperlimits_x[i]

			temp_array=n.ones((size,size))
			for l in range(0,size):
				a=a_range[l]
				for m in range(0,size):
					b=b_range[m]
					y= a*xpoint+b
					if y>ylim:
						### Gaussian above y=ylim
						temp_array[l,m]=gauss(y,1,ylim,ylim)**2 ### (x,a,x0,sigma):
						### Hard limit:
						# temp_array[l,m]=0 # Note that this may be inappropriate for many applications


			prob_array=temp_array*prob_array

			p.fill_between(a_range,a_range*-xpoint+ylim,blims[1],color='k',alpha=0.1)

	p.contour(a_range,b_range,prob_array.T)

	sample=draw_samples(prob_array,n_samples)
	
	ax4=p.subplot(222)
	ax4.set_ylim(blims)
	ax4.set_xlim(alims)
	p.scatter(a_range[sample[:,0]],b_range[sample[:,1]],marker='.')
	new_sample=n.zeros(sample.shape,dtype=float)
	new_sample[:,0]=a_range[sample[:,0]]
	new_sample[:,1]=b_range[sample[:,1]]


	ax2=p.subplot(224)

	popt,pcov = curve_fit(gauss,b_range,n.sum(prob_array,axis=0),p0=[1e-25,35,5])#,p0=[1,mean,sigma])
	p.plot(n.sum(prob_array,axis=0),b_range,color='k')
	p.plot(gauss(b_range, *popt),b_range,color='r')
	b,berr=popt[1],popt[2]
	print 'b =',popt[1],'+/-',popt[2]

	ax3=p.subplot(221)
	p.plot(a_range,n.sum(prob_array,axis=1),color='k')

	print 'P(a>=0) =',sum(n.sum(prob_array,axis=1)[a_range>=0])/n.sum(prob_array)
	print 'P(a<0) =',sum(n.sum(prob_array,axis=1)[a_range<0])/n.sum(prob_array)

	popt,pcov = curve_fit(gauss,a_range,n.sum(prob_array,axis=1),p0=[1e-25,50,5])
	a,aerr=popt[1],popt[2]
	print 'a =',popt[1],'+/-',popt[2]

	# print sample
	p.plot(a_range,gauss(a_range, *popt),color='r')

	p.savefig(stem+'.pdf')

	b=n.mean(new_sample[:,1])
	berr=n.std(new_sample[:,1])
	a=n.mean(new_sample[:,0])
	aerr=n.std(new_sample[:,0])


	return a,b,aerr,berr,new_sample
	# exit()
