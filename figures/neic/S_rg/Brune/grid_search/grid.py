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

def grid_search(**kwargs):
    mr = kwargs.get('mr')
    fce = kwargs.get('fce')
    fcm = kwargs.get('fcm')
    masterid = kwargs.get('masterid')
    egfid = kwargs.get('egfid')
    datapath = kwargs.get('datapath')

    filename = glob.glob('{}/master_{}/egf_{}/S/gcarc_85/all.*'.format(datapath,masterid,egfid))[0]
    data = np.genfromtxt('{}'.format(filename))
    data = data[data[:,0]>0.025]
    data = data[data[:,0]<2.0]

    # grid
    fce_array = np.linspace(-1.6,0.3,80)
    fcm_array = np.linspace(-1.6,0.3,80)
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
    set_range = 1.04
    fcm_min,fce_min,fcm_max,fce_max = find_range(mis=mis,fce_mesh=fce_mesh,fcm_mesh=fcm_mesh,set_range=set_range)

    # bootstrap
    b_fcm = my_bootstrap(mr=mr,fce=fce,fcm=fcm,frq=data[:,0],amp=data[:,1])

    # plot figure
    fig = plt.figure(figsize=[5,5])
    gs = gridspec.GridSpec(2,2)
    ax0 = fig.add_subplot(gs[0,:])
    ax0.plot(data[:,0],data[:,1],lw=0.5,color='k',alpha=0.75)
    ax0.plot(data[:,0],10**func(data[:,0],mr,fce,fcm),lw=1,color='red',alpha=0.75)
    ax0.plot(data[:,0],10**func(data[:,0],mr,fce_min,fcm_min),lw=1,color='orange',alpha=0.75)
    ax0.plot(data[:,0],10**func(data[:,0],mr,fce_max,fcm_max),lw=1,color='green',alpha=0.75)
    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel('Ratio')
    ax0.set_title('{}_{}'.format(masterid,egfid))
    ax1 = fig.add_subplot(gs[1,0])
#    pos = ax1.pcolor(fce_mesh,fcm_mesh,mis,cmap='Blues_r',vmin=0,vmax=5)
    cs = ax1.contourf(fce_mesh,fcm_mesh,mis,levels=[1,1.05,1.2,1.5],corner_mask=False,cmap='Blues_r',linestyles='-',origin='lower',vmin=1,vmax=1.5)
    ax1.plot(np.log10(fce),np.log10(fcm),linestyle='',color='r',marker='o',markersize=1)
    ax1.plot(np.log10(fce_min),np.log10(fcm_min),linestyle='',color='orange',marker='o',markersize=1)
    ax1.plot(np.log10(fce_max),np.log10(fcm_max),linestyle='',color='green',marker='o',markersize=1)
    fig.colorbar(cs)
#    plt.clabel(cs,fmt='%4.2f',colors='k',fontsize=6)
    ax1.set_xlabel('$log_{10}(f_c^{eGf})$')
    ax1.set_ylabel('$log_{10}(f_c^M)$')
    ax1.set_title(r'$misfit<%.3f range:%.3f-%.3f$' % (set_range,np.log10(fcm_min),np.log10(fcm_max)),size=8)
    ax2 = fig.add_subplot(gs[1,1])
    n,bins,patches = ax2.hist(np.log10(b_fcm),bins=50,lw=0.1,density=True)
    (mu,sigma) = norm.fit(np.log10(b_fcm))
    y = norm.pdf(bins,mu,sigma)
    ax2.plot(bins,y,'r--',linewidth=2)
    ax2.set_xlabel('$log_{10}(f_c^M)$')
    ax2.set_ylabel('Density')
    ax2.set_title(r'$\mu=%.3f,\ 2\sigma=%.3f-%.3f$' %(mu,mu-2*sigma,mu+2*sigma),size=8)
    fig.tight_layout()
    plt.savefig('{}_{}.pdf'.format(masterid,egfid))
    plt.close()

def main():
    data = pd.read_csv('pairsfile_rgs_select.csv',skipinitialspace=True)
    data_array = np.array(data)
    datapath = '/home/meichen/work1/SR_Attn/pair_events'

    for i in np.arange(len(data_array[:,0])):
        masterid = data_array[i,0]
        egfid = data_array[i,6]
        mr = data_array[i,15]
        fce = data_array[i,16]
        fcm = data_array[i,17]
        
        grid_search(masterid=masterid,egfid=egfid,mr=mr,fce=fce,fcm=fcm,datapath=datapath)


main()
