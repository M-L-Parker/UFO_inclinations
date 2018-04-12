#!/usr/bin/env python
###########################################################################################

### Neat version

###########################################################################################

import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as p

def r_isco (a):

     ## Innermost (marginally) stable circular orbit (in equatorial plane) around BH of spin a

     z1 = 1 + (1-a**2)**(1./3) * ( (1+a)**(1./3) + (1-a)**(1./3) )
     z2 = ( 3 * a**2 + z1**2 )**0.5

     return 3 + z2 - ( ( 3 - z1 )*( 3 + z1 + 2*z2 ) )**0.5

## (lambda is misspelt to avoid clash with built-in lambda)

def lamba_max (r_psi, Omega_e, theta_o):

     ## Maximum value of conserved AM parameter lambda, subject to reaching theta_o at infinity and passing through r_psi and Omega_e

     a = (r_psi*Omega_e)**2 - np.tan(theta_o)**-2
     b = -2.*r_psi**2*Omega_e
     c = r_psi**2 + np.cos(theta_o)**2

     return (-b-np.sqrt(b**2-4*a*c))/2/a


def g_l_eq_plane (lamba, r, a):

     ## Frequency shift (as g factor) at infinity from radius r in the equatorial plane, for null geodesic with angular momentum l around BH of spin a

     Delta = r**2 + a**2 - 2*r
     A = (r**2 + a**2)**2 - a**2 * Delta
     omega = 2 * a * r / A

     Sigma = r**2
     e2psi = A / Sigma
     e2nu = Delta / e2psi
     enu = np.sqrt(e2nu)

     Omega_e = 1/(r**1.5 + a)

     V_e2 = (Omega_e - omega)**2 * e2psi / e2nu

     return enu * ( 1 - V_e2 )**0.5 / ( 1 - Omega_e*lamba )


def g_eq_plane(r,a,theta_o):

     ## For maximum lambda (hence maximum g) at given r,a,theta_o

     ### Calculated for point of emission ( theta = pi/2, r=r_emis )
     Delta = r**2 + a**2 - 2*r
     A = (r**2 + a**2)**2 - a**2 * Delta
     omega = 2 * a * r / A

     Sigma = r**2
     e2psi = A / Sigma
     e2nu = Delta / e2psi
     enu = np.sqrt(e2nu)

     Omega_e = 1/(r**1.5 + a)

     V_e2 = (Omega_e - omega)**2 * e2psi / e2nu

     return g_l_eq_plane (lamba_max( r/(enu * ( 1 - V_e2 )**0.5), Omega_e, theta_o ), r, a)

def g_max (a,theta, degrees=True):

     ## maximum g factor for accretion disc (extending to isco) of BH with spin a viewed at inclination theta.

     ## Theta is in degrees if degrees==True, otherwise radians

     ## Work in radians:
     if degrees:
         theta_r = theta * np.pi/180
     else:
         theta_r = theta

     ## Maximise g factor across radii

     def obj_func(r):
         return -1*g_eq_plane(r,a,theta_r)

     return -1*op.minimize(obj_func, r_isco(a)*1.1, 
bounds=[(r_isco(a),None)]).fun



def r_of_g_max (a,theta, degrees=True):

     ## radius producing maximum g factor for accretion disc (extending to isco) of BH with spin a viewed at inclination theta.

     ## Theta is in degrees if degrees==True, otherwise radians

     ## Work in radians:
     if degrees:
         theta_r = theta * np.pi/180
     else:
         theta_r = theta

     ## Maximise g factor across radii

     def obj_func(r):
         return -1*g_eq_plane(r,a,theta_r)

     return -1*op.minimize(obj_func, r_isco(a)*1.1).x


if __name__ == '__main__':
    
    ###########################################################################################

    ### Examples

    ###########################################################################################

    ## Maximum blueshift as function of inclination

    incls = np.arange(100)*.9

    a = 0.3

    gs = np.empty(100)

    i=-1
    for incl in incls:
         i+=1

         gs[i] = gs[i] = g_max(a,incl)

    p.plot(incls,gs)
    p.show()


    ## Maximum g as function of radius for different inclinations

    a = 0.998
    a=0.

    radii = np.arange(100)*0.1+r_isco(a)

    incls = np.arange(10)*9 * np.pi/180


    gs = np.empty(100)

    for incl in incls:

         i=-1
         for r in radii:
             i+=1
             gs[i] = g_eq_plane(r,a,incl)

         p.plot(radii,gs)

    p.show()
