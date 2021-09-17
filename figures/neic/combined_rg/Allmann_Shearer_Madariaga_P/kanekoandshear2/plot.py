#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

def M02mag(M0):
    return 2./3.*np.log10(M0)-6.03

def tao2r(tao,M0):
    return (7.*M0/16./tao)**(1./3.)/1000.

P_data_Brune = np.genfromtxt('P_rg_tao_Madariaga_Brune.txt')
S_data_Brune = np.genfromtxt('S_rg_tao_Madariaga_Brune.txt')
AS_data = np.genfromtxt('Allmann_Shearer.txt')
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)
#poli = np.genfromtxt('catalog_poli')
AS_M0 = AS_data[:,0]*1e-7
AS_tao = AS_data[:,1]
AS_fc = AS_data[:,2]
AS_mag = M02mag(AS_M0)
AS_r = []
for i in np.arange(len(AS_tao)):
    AS_r.append(tao2r(AS_tao[i]*1e6,AS_M0[i]))
P_tao_Brune = P_data_Brune[:,1]
P_M0_Brune = P_data_Brune[:,2]
P_lat_Brune = P_data_Brune[:,3]
P_lon_Brune = P_data_Brune[:,4]
P_depth_Brune = P_data_Brune[:,5]
P_mag_Brune = P_data_Brune[:,6]
P_fc_Brune = P_data_Brune[:,7]
P_r_Brune = []
S_tao_Brune = S_data_Brune[:,1]
S_M0_Brune = S_data_Brune[:,2]
S_lat_Brune = S_data_Brune[:,3]
S_lon_Brune = S_data_Brune[:,4]
S_depth_Brune = S_data_Brune[:,5]
S_mag_Brune = S_data_Brune[:,6]
S_fc_Brune = S_data_Brune[:,7]
S_r_Brune = []
for i in np.arange(len(P_tao_Brune)):
    P_r_Brune.append(tao2r(P_tao_Brune[i],P_M0_Brune[i]))
for i in np.arange(len(S_tao_Brune)):
    S_r_Brune.append(tao2r(S_tao_Brune[i],S_M0_Brune[i]))
#poli_tao = []
#for i in np.arange(len(poli[:,0])):
#    if poli[i,1]>400:
#        poli_tao.append(poli[i,5])
#poli_tao = np.array(poli_tao)
#print(np.median(poli_tao*1e-6),np.median(P_tao_Brune*1e-6),np.median(AS_tao))
print(np.mean(np.log10(AS_tao)),np.std(np.log10(AS_tao)))


fig,ax = plt.subplots(1,2,figsize=[8,4])

ax[0].plot(AS_mag,AS_tao,linestyle='',marker='o',markeredgecolor='k',markerfacecolor='none',markersize=5,markeredgewidth=0.5,alpha=0.5)
ax[0].plot(P_mag_Brune,P_tao_Brune*1e-6,linestyle='',marker='o',markeredgecolor='k',markerfacecolor='magenta',markersize=5,markeredgewidth=0.5,alpha=0.65,label='Vr/Vs=0.9')
ax[0].plot(P_mag_Brune,2.5*P_tao_Brune*1e-6,linestyle='',marker='o',markeredgecolor='k',markerfacecolor='orange',markersize=5,markeredgewidth=0.5,alpha=0.65,label='Vr/Vs=0.5')
ax[0].set_yscale('log')
ax[0].set_ylabel(r'$\Delta\tau$ (MPa)',size=14)
ax[0].set_xlabel(r'M$_{\rm W}$',size=14)
ax[0].set_title('(a)',size=14)
ax[0].set_ylim([0.01,20000])
ax[0].set_yticks([0.01,0.1,1,10,100,1000,10000])
ax[0].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[0].tick_params(axis='both',labelsize=10)

ax[1].plot(AS_mag,AS_tao,linestyle='',marker='o',markeredgecolor='k',markerfacecolor='none',markersize=5,markeredgewidth=0.5,alpha=0.5)
ax[1].plot(S_mag_Brune,S_tao_Brune*1e-6,linestyle='',marker='d',markeredgecolor='k',markerfacecolor='blue',markersize=5,markeredgewidth=0.5,alpha=0.65,label='Vr/Vs=0.9')
ax[1].plot(S_mag_Brune,2.5*S_tao_Brune*1e-6,linestyle='',marker='d',markeredgecolor='k',markerfacecolor='green',markersize=5,markeredgewidth=0.5,alpha=0.65,label='Vr/Vs=0.5')
ax[1].set_yscale('log')
ax[1].set_ylabel(r'$\Delta\tau$ (MPa)',size=14)
ax[1].set_xlabel(r'M$_{\rm W}$',size=14)
ax[1].set_title('(b)',size=14)
ax[1].set_ylim([0.01,20000])
ax[1].set_yticks([0.01,0.1,1,10,100,1000,10000])
ax[1].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[1].tick_params(axis='both',labelsize=10)
ax[0].legend()
ax[1].legend()
fig.tight_layout()
fig.savefig('tao.pdf')
