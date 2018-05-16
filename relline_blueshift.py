#!/usr/bin/env python

import numpy as np
from xspec import *

AllModels.lmod("relxill","/Users/mlparker/programs/xspec/relxill/")

mod=Model('relline')


Plot.device = "/xs"
Plot.xAxis='keV'
Plot("eem")

spins=np.linspace(0,0.998,50)
inclinations=np.linspace(3,89,50)

rest_E=6.4

# step over grid of a and i values
meta_list=[]
for spin in spins:
	temp_list=[]
	mod.setPars({5:spin})

	for inc in inclinations:
		
		mod.setPars({6:inc})
		Plot()

		# x and y values from the plot
		xs=np.array(Plot.x())
		ys=np.array(Plot.model())

		# Max blueshift corresponds to last energy where relline>0
		max_x=xs[ys>0][-1]

		# convert to velocity:
		max_v=((max_x/rest_E)**2-1)/((max_x/rest_E)**2+1)

		temp_list.append(max_v)

	meta_list.append(temp_list)

velocity_array=np.array(meta_list)

np.savetxt('relline_velocity_array.dat',velocity_array)