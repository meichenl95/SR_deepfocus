#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
import pandas as pd
import os
import glob
import subprocess
from scipy.stats import norm

def func(x,a,b,c):
    return np.log10(a) + np.log10(1 + x**2/b**2) - np.log10(1+ x**2/c**2)

def misfit(**kwargs):
    mr = kwargs.get('mr')
    grid_fce = kwargs.get('grid_fce')
    grid_fcm = kwargs.get('grid_fcm')
    frq = kwargs.get('frq')
    amp = kwargs.get('amp')

    syn = func(frq,mr,grid_fce,grid_fcm)
    # huber scaled to 0.1
    mis = 0
    residual = abs(syn-np.log10(amp))
    for i in np.arange(len(residual)):
        if residual[i] < 0.1:
            mis = mis + residual[i]**2
        else:
            mis = mis + 0.2*residual[i] - 0.01
    mis = mis/len(residual)

    return mis
    
def find_range(**kwargs):
    mis = kwargs.get('mis')
    fcm_mesh = kwargs.get('fcm_mesh')
    fce_mesh = kwargs.get('fce_mesh')
    set_range = kwargs.get('set_range')

    fce_list = []
    fcm_list = []
    for row in np.arange(len(mis[:,0])):
        for col in np.arange(len(mis[0,:])):
            if mis[row,col] < set_range:
                fce_list.append(fce_mesh[row,col])
                fcm_list.append(fcm_mesh[row,col])

    index_min = fcm_list.index(np.min(fcm_list))
    index_max = fcm_list.index(np.max(fcm_list))

    return 10**fcm_list[index_min],10**fce_list[index_min],10**fcm_list[index_max],10**fce_list[index_max]

def my_bootstrap(**kwargs):
    mr = kwargs.get('mr')
    fce = kwargs.get('fce')
    fcm = kwargs.get('fcm')
    frq = kwargs.get('frq')
    amp = kwargs.get('amp')

    res = np.log10(amp) - func(frq,mr,fce,fcm)
    b_fcm = []
    for i in np.arange(1000):
        random_index = np.random.randint(0,len(amp),size=len(amp))
        new_y = func(frq,mr,fce,fcm) + [res[j] for j in random_index]
        try:
            popt,pcov = curve_fit(func,frq,new_y,bounds=([1.,0.,0.],[100000,100.,100.]),method='trf',loss='huber',f_scale=0.1)
            b_fcm.append(popt[2])
        except:
            print('Error - curve_fit failed')
    return np.array(b_fcm)


def main():
    data = pd.read_csv('pairsfile_rgs_select.csv',skipinitialspace=True)
    data_array = np.array(data)
    datapath = '/home/meichen/work1/SR_Attn/pair_events'

    fig = plt.figure(figsize=[5,8])
    gs = gridspec.GridSpec(4,2)
    ax0 = fig.add_subplot(gs[0,:])
    for n,color,cmap,line in zip(np.arange(len(data_array[:,0])),['blue','red','green'],['Blues_r','Reds_r','Greens_r'],[2,3,1]):
        masterid = data_array[n,0]
        egfid = data_array[n,6]
        mr = data_array[n,15]
        fce = data_array[n,16]
        fcm = data_array[n,17]
        
        filename = glob.glob('{}/master_{}/egf_{}/S/gcarc_85/all.*'.format(datapath,masterid,egfid))[0]
        data = np.genfromtxt('{}'.format(filename))
        data = data[data[:,0]>0.025]
        data = data[data[:,0]<2.0]
    
        # grid
        fce_array = np.linspace(np.log10(fce)-0.15,np.log10(fce)+0.15,160)
        fcm_array = np.linspace(np.log10(fcm)-0.15,np.log10(fcm)+0.15,160)
        fce_mesh,fcm_mesh = np.meshgrid(fce_array,fcm_array)
        mis = np.zeros((len(fcm_array),len(fcm_array)))
        for i in np.arange(len(fcm_array)):
            for j in np.arange(len(fcm_array)):
                grid_fce = 10**fce_mesh[i,j]
                grid_fcm = 10**fcm_mesh[i,j]
                mis[i,j] = misfit(mr=mr,grid_fce=grid_fce,grid_fcm=grid_fcm,frq=data[:,0],amp=data[:,1])
    
        # relative misfit
        mis = mis/np.min(mis)
    
        # find range
        set_range = 1.01
        fcm_min,fce_min,fcm_max,fce_max = find_range(mis=mis,fce_mesh=fce_mesh,fcm_mesh=fcm_mesh,set_range=set_range)
    
        # bootstrap
        b_fcm = my_bootstrap(mr=mr,fce=fce,fcm=fcm,frq=data[:,0],amp=data[:,1])
    
        # plot figure
        ax0.plot(data[:,0],data[:,1],lw=0.5,color=color,alpha=0.75)
        ax0.plot(data[:,0],10**(func(data[:,0],mr,fce,fcm)),color='gray',linestyle='--',lw=1,alpha=0.75)
        ax0.plot(fcm,10**func(fcm,mr,fce,fcm)+0.5,marker='v',markeredgewidth=2,markeredgecolor=color)
        ax2 = fig.add_subplot(gs[line,1])
        n,bins,patches = ax2.hist(np.log10(b_fcm),bins=50,edgecolor='gray',facecolor='gray',lw=0.1,alpha=0.75)
        (mu,sigma) = norm.fit(np.log10(b_fcm))
        y = norm.pdf(bins,mu,sigma)*1000*(bins[1]-bins[0])
        ax2.plot(bins,y,color=color,linestyle='--',linewidth=2)
        ax2.set_xlabel(r'log$_{\rm 10}$(f$_{\rm M}$)',size=8)
        ax2.set_ylabel('Counts',size=8)
        ax2.text(0.05,0.8,'$\mu$ = %.3f\n$2\sigma$ = %.3f' % (mu,2*sigma),transform=ax2.transAxes,fontsize=6)
        ax2.tick_params(axis='both',labelsize=6)
#        ax2.set_xlim([np.min(np.log10(b_fcm)),np.max(np.log10(b_fcm))])
        ax1 = fig.add_subplot(gs[line,0])
        cs = ax1.contourf(fcm_mesh,fce_mesh,mis,levels=[1,1.01,1.02,1.03],corner_mask=False,cmap=cmap,linestyles='-',origin='lower',vmin=1,vmax=1.03)
        cbar = fig.colorbar(cs)
        cbar.ax.tick_params(labelsize=6)
        ax1.set_ylabel(r'log$_{\rm 10}$(f$_{\rm eGf}$)',size=8)
        ax1.set_xlabel(r'log$_{\rm 10}$(f$_{\rm M}$)',size=8)
        ax1.set_xlim([np.log10(fcm)-4*sigma,np.log10(fcm)+4*sigma])
        ax1.set_ylim([np.log10(fce)-4*sigma,np.log10(fce)+4*sigma])
        ax1.text(0.05,0.9,'[%.3f,%.3f]' % (np.log10(fcm_min),np.log10(fcm_max)),transform=ax1.transAxes,fontsize=6)
        ax1.text(0.75,0.05,'eGf {}'.format(line),transform=ax1.transAxes,fontsize=8,color=color)
        ax1.tick_params(axis='both',labelsize=6)

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel('Frequency (Hz)',size=8)
    ax0.set_ylabel('Spectral ratios',size=8)
    ax0.set_xticks([0.025,0.05,0.1,0.5,1,2])
    ax0.set_xticklabels([0.025,0.05,0.1,0.5,1,2])
    ax0.set_yticks([20,100,900])
    ax0.set_yticklabels([20,100,900])
    ax0.tick_params(axis='both',labelsize=6)
    ax0.text(0.0,1.05,'(a)',fontsize=8,transform=ax0.transAxes)
    ax0.text(0.0,-0.37,'(b)',fontsize=8,transform=ax0.transAxes)
    ax0.text(0.57,-0.37,'(c)',fontsize=8,transform=ax0.transAxes)
    fig.tight_layout()
    plt.savefig('{}.pdf'.format(masterid))
    plt.close()
main()
