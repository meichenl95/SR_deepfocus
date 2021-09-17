#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from sys import argv
from scipy.optimize import curve_fit

def func(x,a,b,c):
    return np.log10(a) + np.log10(1 + x**2/b**2) - np.log10(1 + x**2/c**2)

def fit(x,y):
    popt,pcov = curve_fit(func,x,y,bounds=([1.0,0.0,0.0],[100000,100.0,100.0]),method='trf',loss='huber',f_scale=0.1)
    return popt[0],popt[1],popt[2]

def fit_fce(x,y,**kwargs):
    c = kwargs.get('c')
    try:
        popt,pcov = curve_fit(lambda x,a,b:func(x,a,b,c),x,y,bounds=([1.0,c],[100000,100.0]),method='trf',loss='huber',f_scale=0.1)
        print(popt)
        return popt[0],popt[1]
    except:
        return 0,0

def main(**kwargs):

    filename = kwargs.get('filename')
    phase = kwargs.get('phase')
    distance = kwargs.get('distance')

    data = pd.read_csv('{}'.format(filename),skipinitialspace=True)
    data_array = np.array(data)
    current_path = '/home/meichen/Research/SR_Attn/pair_events/figures/egf_combine'
    file_path = '/home/meichen/work1/SR_Attn/pair_events'
    
    pairs = {}
    for i in np.arange(len(data_array[:,0])):
        if data_array[i,10] <3.0:
            pairs.setdefault(data_array[i,0],[]).append(data_array[i,6])
            srname = glob.glob('{}/master_{}/egf_{}/{}/gcarc_{}/all*'.format(file_path,data_array[i,0],data_array[i,6],phase,distance))[0]
            d = np.genfromtxt('{}'.format(srname))
            d[:,1] = d[:,1]/data_array[i,15]
            np.savetxt('{}/{}_{}_{}.txt'.format(current_path,data_array[i,0],data_array[i,6],phase),d)

    masterid_list = []
    fcm_list = []
    for key in list(pairs.keys()):
        freq = np.zeros(1001)
        amp = np.zeros(1001)
        n = 0
        for value in list(pairs.get(key)):
            d = np.genfromtxt('{}_{}_{}.txt'.format(key,value,phase))
            freq = d[:,0]
            amp = amp + d[:,1]
            n = n + 1
        amp = amp/n
        mr,fce,fcm = fit(freq, np.log10(amp))
        if n >5:
            masterid_list.append(key)
            fcm_list.append(fcm)        

        fig,ax = plt.subplots(1,1,figsize=[6,3])
        ax.plot(freq,amp)
        ax.plot(freq,10**(func(freq, mr,fce,fcm)),linestyle='--',color='grey',lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amp')
        ax.set_title('{}_{}'.format(key,phase))
        fig.tight_layout()
        plt.savefig('{}_{}.png'.format(key,phase))
        plt.close()
    np.savetxt('master_fc_{}.out'.format(phase),np.vstack((masterid_list,fcm_list)).T,fmt='%d %f')

    m_id = []
    e_id = []
    fce_list = []
    fce_original = []
    for i in np.arange(len(masterid_list)):
        key = masterid_list[i]
        fcm = fcm_list[i]
        for value in pairs.get(key):
            d = np.genfromtxt('{}_{}_{}.txt'.format(key,value,phase))
            mr,fce = fit_fce(d[:,0],np.log10(d[:,1]),c=fcm)
            if fce < 0.67:
                m_id.append(key)
                e_id.append(value)
                fce_list.append(fce)

                for j in np.arange(len(data_array[:,0])):
                    if data_array[j,0] == key and data_array[j,6] == value:
                        fce_original.append(data_array[j,16])

                fig,ax = plt.subplots(1,1,figsize=[6,3])
                ax.plot(d[:,0],d[:,1])
                ax.plot(d[:,0],10**(func(d[:,0],mr,fce,fcm)),linestyle='--',color='gray',lw=1)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Amp')
                ax.set_title('{}_{}_{}'.format(key,value,phase))
                fig.tight_layout()
                plt.savefig('{}_{}_{}.png'.format(key,value,phase))
                plt.close()
    np.savetxt('egf_fc_{}.out'.format(phase),np.vstack((m_id,e_id,fce_list,fce_original)).T,fmt='%d %d %f %f')

    # figure
    fig,ax = plt.subplots(1,1,figsize=[4,4])
    ax.plot(fce_list,fce_original,linestyle='',marker='o',mec='k',mfc='k',markersize=2)
    ax.set_xlabel('fce_constrained')
    ax.set_ylabel('fce_original')
    ax.set_xlim([0,0.8])
    ax.set_ylim([0,0.8])
    ax.set_title('{}'.format(phase))
    fig.tight_layout()
    plt.savefig('fce_{}.png'.format(phase))
    plt.close()

main(filename=argv[1],phase=argv[2],distance='85') 
