#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

P_data = np.genfromtxt('P_rg_tao.txt')
S_data = np.genfromtxt('S_rg_tao.txt')

fig,ax = plt.subplots(2,1,figsize=[6,4])
ax[0].plot(S_data[:,5],S_data[:,1]*1e-6,linestyle='',marker='d',markeredgecolor='k',markeredgewidth=0.5,markersize=5,markerfacecolor='blue',alpha=0.75)
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\Delta\tau_{\rm S}$ (MPa)',size=10)
ax[0].tick_params(axis='y',which='minor',labelleft=False)
ax[0].set_yticks([0.1,1,10,100,1000,10000,20000])
ax[0].set_yticklabels([0.1,1,10,100,1000,10000,20000])
ax[0].tick_params(axis='both',labelsize=6)
ax[1].plot(P_data[:,5],P_data[:,1]*1e-6,linestyle='',marker='o',markeredgecolor='k',markeredgewidth=0.5,markersize=5,markerfacecolor='magenta',alpha=0.75)
print(np.max(P_data[:,1]*1e-6))
ax[1].set_yscale('log')
ax[1].set_xlabel('Depth (km)')
ax[1].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=10)
ax[1].tick_params(axis='y',which='minor',labelleft=False)
ax[1].set_yticks([0.1,1,10,100,1000,10000,20000])
ax[1].set_yticklabels([0.1,1,10,100,1000,10000,20000])
ax[1].tick_params(axis='both',labelsize=6)
plt.savefig('result.pdf')

# Fiji Tonga region
tao_list = []
depth_list = []
for i in np.arange(len(S_data[:,0])):
    if S_data[i,4] > 174 or S_data[i,4] < -173:
        if S_data[i,3] > -25 and S_data[i,3] < -15:
            tao_list.append(S_data[i,1])
            depth_list.append(S_data[i,5])
fig,ax = plt.subplots(2,1,figsize=[7,5])
ax[0].plot(depth_list,tao_list,linestyle='',marker='d',markeredgecolor='k',markeredgewidth=1,markersize=7,markerfacecolor='blue',alpha=0.75)
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\Delta\tau_S$ (MPa)')
tao_list = []
depth_list = []
for i in np.arange(len(P_data[:,0])):
    if P_data[i,4] > 174 or P_data[i,4] < -173:
        if P_data[i,3] > -25 and P_data[i,3] < -15:
            tao_list.append(P_data[i,1])
            depth_list.append(P_data[i,5])
ax[1].plot(depth_list,tao_list,linestyle='',marker='o',markeredgecolor='k',markeredgewidth=1,markersize=7,markerfacecolor='magenta',alpha=0.75)
ax[1].set_yscale('log')
ax[1].set_ylabel(r'$\Delta\tau_P$ (MPa)')
plt.savefig('Fiji.pdf')

