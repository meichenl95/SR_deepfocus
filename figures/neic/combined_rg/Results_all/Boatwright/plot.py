#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

P_data = np.genfromtxt('P_rg_tao.txt')
S_data = np.genfromtxt('S_rg_tao.txt')
exmp = np.genfromtxt('example_P.txt')
exms = np.genfromtxt('example_S.txt')
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)

fig,ax = plt.subplots(2,2,figsize=[7,5])
# corner frequency VS. magnitude
ax[0,1].errorbar(S_data[:,6],S_data[:,7],yerr=np.vstack((S_data[:,8],S_data[:,9])),linestyle='',marker='d',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='deepskyblue',alpha=0.75)
ax[0,1].set_yscale('log')
ax[0,1].set_ylabel(r'f$_{\rm c}^{\rm S}$ (Hz)',size=10)
ax[0,1].tick_params(axis='y',which='minor',labelleft=False)
ax[0,1].set_ylim([0.05,0.7])
ax[0,1].set_xlim([6.3,8.4])
ax[0,1].set_yticks([0.05,0.1,0.5,0.7])
ax[0,1].set_yticklabels([0.05,0.1,0.5,0.7])
ax[0,1].tick_params(axis='both',labelsize=6)
ax[0,1].text(0.05,0.9,'(b)',transform=ax[0,1].transAxes,fontsize=10)
ax[0,1].text(0.75,0.05,'S wave',transform=ax[0,1].transAxes,fontsize=10,color='deepskyblue')
ax[0,1].errorbar(exms[:,6],exms[:,7],yerr=np.vstack((exms[:,8],exms[:,9])),linestyle='',marker='d',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='k',alpha=0.75)
ax[0,0].errorbar(P_data[:,6],P_data[:,7],yerr=np.vstack((P_data[:,8],P_data[:,9])),linestyle='',marker='o',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='magenta',alpha=0.75)
ax[0,0].errorbar(exmp[:,6],exmp[:,7],yerr=np.vstack((exmp[:,8],exmp[:,9])),linestyle='',marker='o',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='k',alpha=0.75)
for i in np.arange(len(exmp[:,0])):
    ax[0,0].text(exmp[i,6]-0.15,exmp[i,7],' %d' % exmp[i,13],fontsize=6)
    ax[0,1].text(exms[i,6]-0.15,exms[i,7],' %d' % exms[i,13],fontsize=6)
    ax[1,0].text(exmp[i,6]-0.15,exmp[i,1]*1e-6,' %d' % exmp[i,13],fontsize=6)
    ax[1,1].text(exms[i,6]-0.15,exms[i,1]*1e-6,' %d' % exms[i,13],fontsize=6)
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel(r'f$_{\rm c}^{\rm P}$ (Hz)',size=10)
ax[0,0].set_yticks([0.05,0.1,0.5,0.7])
ax[0,0].set_yticklabels([0.05,0.1,0.5,0.7])
ax[0,0].tick_params(axis='both',labelsize=6)
ax[0,0].set_ylim([0.05,0.7])
ax[0,0].set_xlim([6.3,8.4])
ax[0,0].text(0.05,0.9,'(a)',transform=ax[0,0].transAxes,fontsize=10)
ax[0,0].text(0.75,0.05,'P wave',transform=ax[0,0].transAxes,fontsize=10,color='magenta')
# stress drop VS. magnitude
ax[1,1].errorbar(S_data[:,6],S_data[:,1]*1e-6,yerr=np.vstack((S_data[:,10]*1e-6,S_data[:,11]*1e-6)),linestyle='',marker='d',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='deepskyblue',alpha=0.75)
ax[1,1].errorbar(exms[:,6],exms[:,1]*1e-6,yerr=np.vstack((exms[:,10]*1e-6,exms[:,11]*1e-6)),linestyle='',marker='d',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='k',alpha=0.75)
ax[1,1].set_yscale('log')
ax[1,1].set_ylabel(r'$\Delta\tau_{\rm S}$ (MPa)',size=10)
ax[1,1].hlines(np.median(S_data[:,1])*1e-6,xmin=6.4,xmax=8.3,linestyles='--',color='deepskyblue')
ax[1,1].set_yticks([0.1,1,10,100,1000,10000,20000])
ax[1,1].set_yticklabels([0.1,1,10,100,1000,10000,20000])
ax[1,1].tick_params(axis='both',labelsize=6)
ax[1,1].set_ylim([0.1,20000])
ax[1,1].set_xlim([6.3,8.4])
ax[1,1].set_xlabel(r'M$_{\rm W}$',size=10)
ax[1,1].text(0.05,0.9,'(d)',transform=ax[1,1].transAxes,fontsize=10)
ax[1,1].text(0.75,0.05,'S wave',transform=ax[1,1].transAxes,fontsize=10,color='deepskyblue')
ax[1,0].errorbar(P_data[:,6],P_data[:,1]*1e-6,yerr=np.vstack((P_data[:,10]*1e-6,P_data[:,11]*1e-6)),linestyle='',marker='o',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='magenta',alpha=0.75)
ax[1,0].errorbar(exmp[:,6],exmp[:,1]*1e-6,yerr=np.vstack((exmp[:,10]*1e-6,exmp[:,11]*1e-6)),linestyle='',marker='o',markeredgecolor='k',markeredgewidth=0.5,ecolor='k',elinewidth=0.5,markersize=5,markerfacecolor='k',alpha=0.75)
ax[1,0].set_yscale('log')
ax[1,0].set_xlabel(r'M$_{\rm W}$',size=10)
ax[1,0].set_ylabel(r'$\Delta\tau_{\rm P}$ (MPa)',size=10)
ax[1,0].hlines(np.median(P_data[:,1])*1e-6,xmin=6.4,xmax=8.3,linestyles='--',color='magenta')
ax[1,0].set_yticks([0.1,1,10,100,1000,10000,20000])
ax[1,0].set_yticklabels([0.1,1,10,100,1000,10000,20000],size=6)
ax[1,0].tick_params(axis='both',labelsize=6)
ax[1,0].set_ylim([0.1,20000])
ax[1,0].set_xlim([6.3,8.4])
ax[1,0].text(0.05,0.9,'(c)',transform=ax[1,0].transAxes,fontsize=10)
ax[1,0].text(0.75,0.05,'P wave',transform=ax[1,0].transAxes,fontsize=10,color='magenta')
fig.tight_layout()
print('S mean: %.3f; P mean: %.3f' %(np.mean(np.log10(S_data[:,1]*1e-6)),np.mean(np.log10(P_data[:,1]*1e-6))))
print('S std: %.3f; P std: %.3f' %(np.std(np.log10(S_data[:,1]*1e-6)),np.std(np.log10(P_data[:,1]*1e-6))))
print('S median: %.3f; P median: %.3f' %(np.median(np.log10(S_data[:,1]*1e-6)),np.median(np.log10(P_data[:,1]*1e-6))))
print('Max S:{}, P:{}'.format(np.max(S_data[:,1])*1e-6,np.max(P_data[:,1])*1e-6))
print('Min S:{}, P:{}'.format(np.min(S_data[:,1])*1e-6,np.min(P_data[:,1])*1e-6))


plt.savefig('result.pdf')