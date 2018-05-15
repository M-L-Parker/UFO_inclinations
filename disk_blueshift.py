#!/usr/bin/env python
###########################################################################################

### Neat version

###########################################################################################

import numpy as np
from scipy import optimize as op

def r_isco (a):

     z1 = 1 + (1-a**2)**(1./3) * ( (1+a)**(1./3) + (1-a)**(1./3) )

     z2 = ( 3 * a**2 + z1**2 )**0.5

     return 3 + z2 - np.sign(a) * ( ( 3 - z1 )*( 3 + z1 + 2*z2 ) )**0.5

def doppler_shift(v,theta):
     return 1/np.sqrt(1-v**2)*(1+v*np.cos(theta))

def g_z(r,a):
     Delta = r**2+a**2-2.*r
     Sigma = np.sqrt((r**2+a**2)**2 - a**2*Delta)
     return r/Sigma*np.sqrt(Delta)

def v_circ(r,a):
     # Bardeen 1972
     Delta = r**2+a**2-2.*r
     return (r**2-2.*a*r**0.5+a**2)/Delta**0.5/(r**1.5+a)

def g_max(theta,a, degrees=True):
     ## maximum g factor for accretion disc (extending to isco) of BH with spin a viewed at inclination theta.

     ## Theta is in degrees if degrees==True, otherwise radians

     ## Work in radians:
     if degrees:
         theta_r = theta * np.pi/180
     else:
         theta_r = theta

     ## Maximise g factor across radii

     def obj_func(r):
         return -1*g_z(r,a)*doppler_shift(v_circ(r,a),np.pi/2-theta_r)

     return -1*op.minimize(obj_func, r_isco(a)*3.1, bounds=[(r_isco(a), None)], tol=1e-8).fun

def r_of_g_max(theta,a, degrees=True):
     ## maximum g factor for accretion disc (extending to isco) of BH with spin a viewed at inclination theta.

     ## Theta is in degrees if degrees==True, otherwise radians

     ## Work in radians:
     if degrees:
         theta_r = theta * np.pi/180
     else:
         theta_r = theta

     ## Maximise g factor across radii

     def obj_func(r):
         return -1*g_z(r,a)*doppler_shift(v_circ(r,a),np.pi/2-theta_r)

     return op.minimize(obj_func, r_isco(a)*3.1, bounds=[(r_isco(a), None)], tol=1e-8).x
