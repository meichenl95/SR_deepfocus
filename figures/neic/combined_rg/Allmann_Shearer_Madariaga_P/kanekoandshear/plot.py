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
for i in np.arange(len(P_tao_Brune)):
    P_r_Brune.append(tao2r(P_tao_Brune[i],P_M0_Brune[i]))
S_tao_Brune = S_data_Brune[:,1]
S_M0_Brune = S_data_Brune[:,2]
S_lat_Brune = S_data_Brune[:,3]
S_lon_Brune = S_data_Brune[:,4]
S_depth_Brune = S_data_Brune[:,5]
S_mag_Brune = S_data_Brune[:,6]
S_fc_Brune = S_data_Brune[:,7]
S_r_Brune = []
for i in np.arange(len(S_tao_Brune)):
    S_r_Brune.append(tao2r(S_tao_Brune[i],S_M0_Brune[i]))
#poli_tao = []
#for i in np.arange(len(poli[:,0])):
#    if poli[i,1]>400:
#        poli_tao.append(poli[i,5])
#poli_tao = np.array(poli_tao)
#print(np.median(poli_tao*1e-6),np.median(P_tao_Brune*1e-6),np.median(AS_tao))
print(np.mean(np.log10(AS_tao)),np.std(np.log10(AS_tao)))


fig,ax = plt.subplots(2,2,figsize=[9,8])

weights = np.ones_like(np.log10(AS_tao))/float(len(AS_tao))
n, bins, patches = ax[0,0].hist(np.log10(AS_tao),bins=15,edgecolor='gray',facecolor='gray',lw=0.1,alpha=0.5,orientation='horizontal',weights=weights)
n, bins, patches = ax[0,1].hist(np.log10(AS_tao),bins=15,edgecolor='gray',facecolor='gray',lw=0.1,alpha=0.5,orientation='horizontal',weights=weights)
n, bins, patches = ax[1,0].hist(np.log10(AS_tao),bins=15,edgecolor='gray',facecolor='gray',lw=0.1,alpha=0.5,orientation='horizontal',weights=weights)
n, bins, patches = ax[1,1].hist(np.log10(AS_tao),bins=15,edgecolor='gray',facecolor='gray',lw=0.1,alpha=0.5,orientation='horizontal',weights=weights)
(mu,sigma) = norm.fit(np.log10(AS_tao))
y = norm.pdf(np.linspace(-2,3,100), mu, sigma)*np.max(n)*np.sqrt(2*np.pi*sigma**2)
ax[0,0].plot(y,np.linspace(-2,3,100),color='k',linestyle='--',linewidth=1)
ax[0,0].set_ylim([-2,np.log10(20000)])
ax[0,0].set_yticks([-2,-1,0,1,2,3,4])
ax[0,0].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[0,0].set_xlabel('Fraction',size=14)
ax[0,0].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=14)
ax[0,0].set_title('(a)',size=14)
ax[0,0].text(0.8,0.03,'P wave',transform=ax[0,0].transAxes,fontsize=10,color='magenta')

ax[1,0].plot(y,np.linspace(-2,3,100),color='k',linestyle='--',linewidth=1)
ax[1,0].set_ylim([-2,np.log10(20000)])
ax[1,0].set_yticks([-2,-1,0,1,2,3,4])
ax[1,0].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[1,0].set_xlabel('Fraction',size=14)
ax[1,0].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=14)
ax[1,0].set_title('(b)',size=14)
ax[1,0].text(0.8,0.03,'P wave',transform=ax[1,0].transAxes,fontsize=10,color='orange')
#weights = np.ones_like(np.log10(poli_tao*1e-6))/float(len(poli_tao))
#n, bins, patches = ax[2].hist(np.log10(poli_tao*1e-6),bins=10,edgecolor='gray',facecolor='blue',lw=0.1,alpha=0.35,orientation='horizontal',weights=weights)
#(mu,sigma) = norm.fit(np.log10(poli_tao*1e-6))
#y = norm.pdf(np.linspace(-1,3,100),mu,sigma)*np.max(n)*np.sqrt(2*np.pi*sigma**2)
#ax[2].plot(y,np.linspace(-1,3,100),color='blue',linestyle='--',linewidth=1)
weights = np.ones_like(np.log10(P_tao_Brune*1e-6))/float(len(P_tao_Brune))
ax[0,0].hist(np.log10(P_tao_Brune*1e-6),bins=10,edgecolor='gray',facecolor='magenta',lw=0.1,alpha=0.35,orientation='horizontal',weights=weights,label='Vr/Vs=0.9')
ax[1,0].hist(np.log10(P_tao_Brune*2.5*1e-6),bins=10,edgecolor='gray',facecolor='orange',lw=0.1,alpha=0.35,orientation='horizontal',weights=weights,label='Vr/Vs=0.5')
weights = np.ones_like(np.log10(S_tao_Brune*1e-6))/float(len(S_tao_Brune))
ax[0,1].hist(np.log10(S_tao_Brune*1e-6),bins=10,edgecolor='gray',facecolor='blue',lw=0.1,alpha=0.35,orientation='horizontal',weights=weights,label='Vr/Vs=0.9')
ax[1,1].hist(np.log10(S_tao_Brune*1.7*1e-6),bins=10,edgecolor='gray',facecolor='green',lw=0.1,alpha=0.35,orientation='horizontal',weights=weights,label='Vr/Vs=0.5')

ax[0,1].plot(y,np.linspace(-2,3,100),color='k',linestyle='--',linewidth=1)
ax[0,1].set_ylim([-2,np.log10(20000)])
ax[0,1].set_yticks([-2,-1,0,1,2,3,4])
ax[0,1].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[0,1].set_xlabel('Fraction',size=14)
ax[0,1].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=14)
ax[0,1].set_title('(c)',size=14)
ax[0,1].text(0.8,0.03,'S wave',transform=ax[0,1].transAxes,fontsize=10,color='blue')

ax[1,1].plot(y,np.linspace(-2,3,100),color='k',linestyle='--',linewidth=1)
ax[1,1].set_ylim([-2,np.log10(20000)])
ax[1,1].set_yticks([-2,-1,0,1,2,3,4])
ax[1,1].set_yticklabels([0.01,0.1,1,10,100,1000,10000])
ax[1,1].set_xlabel('Fraction',size=14)
ax[1,1].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=14)
ax[1,1].set_title('(d)',size=14)
ax[1,1].text(0.8,0.03,'S wave',transform=ax[1,1].transAxes,fontsize=10,color='green')

ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()

fig.tight_layout()
fig.savefig('tao.pdf')
