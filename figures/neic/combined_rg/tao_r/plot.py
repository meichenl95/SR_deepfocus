#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def tao2r(tao,M0):
    return (7.*M0/16./tao)**(1./3.)/1000.

P_data = np.genfromtxt('P_rg_tao.txt')
S_data = np.genfromtxt('S_rg_tao.txt')
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)

P_r = []
for i in np.arange(len(P_data)):
    P_r.append(tao2r(P_data[i,1],P_data[i,2]))

S_r = []
for i in np.arange(len(S_data)):
    S_r.append(tao2r(S_data[i,1],S_data[i,2]))

fig,ax = plt.subplots(2,1,figsize=[7,5])
ax[0].plot(S_data[:,6],S_r,linestyle='',marker='d',markeredgecolor='k',markeredgewidth=1,markersize=7,markerfacecolor='blue',alpha=0.75)
ax[0].set_ylabel('radius (km)')
ax[1].plot(P_data[:,6],P_r,linestyle='',marker='o',markeredgecolor='k',markeredgewidth=1,markersize=7,markerfacecolor='magenta',alpha=0.75)
ax[1].set_xlabel('Mw')
ax[1].set_ylabel('radius (km)')
plt.savefig('result.pdf')

