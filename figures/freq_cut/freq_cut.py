#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from sys import argv
from scipy.optimize import curve_fit
import subprocess

def func(x,a,b,c):
    return np.log10(a) + np.log10(1 + x**2/b**2) - np.log10(1+ x**2/c**2)

def fit_lower(x,y,fc_lower):
    x_cut = []
    y_cut = []
    for i in np.arange(len(x)):
        if x[i]>fc_lower:
            x_cut.append(x[i])
            y_cut.append(y[i])
    x_cut = np.array(x_cut)
    y_cut = np.array(y_cut)

    try:
        popt,pcov = curve_fit(func, x_cut,y_cut,bounds=([1,0.,0.],[100000,10.,40.]),method='trf',loss='huber',f_scale=0.1)
        return popt[0],popt[1],popt[2]
    except:
        return 0.02,0.02,0.02


def fit_upper(x,y,fc_upper):
    x_cut = []
    y_cut = []
    for i in np.arange(len(x)):
        if x[i]<fc_upper:
            x_cut.append(x[i])
            y_cut.append(y[i])
    x_cut = np.array(x_cut)
    y_cut = np.array(y_cut)

    try:
        popt,pcov = curve_fit(func, x_cut,y_cut,bounds=([1,0.,0.],[100000,10.,40.]),method='trf',loss='huber',f_scale=0.1)
        return popt[0],popt[1],popt[2]
    except:
        return 0.02,0.02,0.02


def main(**kwargs):
    
    filename = kwargs.get('filename')
    phase = kwargs.get('phase')
    distance = kwargs.get('distance')

    data = pd.read_csv('{}'.format(filename),skipinitialspace=True)
    data_array = np.array(data)
    current_path = '/home/meichen/Research/SR_Attn/pair_events/figures/freq_cut'
    file_path = '/home/meichen/work1/SR_Attn/pair_events'

    for i in np.arange(len(data_array[:,0])):
        name = glob.glob('{}/master_{}/egf_{}/{}/gcarc_{}/all.*'.format(file_path,data_array[i,0],data_array[i,6],phase,distance))[0]
        subprocess.call(['cp {} {}_{}_{}.sr'.format(name,data_array[i,0],data_array[i,6],phase)],shell=True)


    fc_lower = 10**np.linspace(-1.6,-1.22,50)
    fc_upper = 10**np.linspace(0.52,0.9,50)
    for i in np.arange(len(data_array[:,0])):
        d = np.genfromtxt('{}_{}_{}.sr'.format(data_array[i,0],data_array[i,6],phase))
        mr_est = []
        fce_est = []
        fcm_est = []
        for j in np.arange(50):
            a,b,c = fit_lower(d[:,0],np.log10(d[:,1]),fc_lower[j])
            mr_est.append(a)
            fce_est.append(b)
            fcm_est.append(c)
        fig,ax = plt.subplots(2,1,figsize=[5,4])
        ax[0].scatter(fc_lower,fcm_est,s=2)
        ax[0].set_ylim([0.001,4])
        ax[0].set_xlabel('fc_lower')
        ax[0].set_ylabel('fit fc')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        if max(np.log10(fcm_est))-min(np.log10(fcm_est)) > 0.5:
            print('{}_{}_{}'.format(data_array[i,0],data_array[i,6],phase))

        mr_est = []
        fce_est = []
        fcm_est = []
        for j in np.arange(50):
            a,b,c = fit_upper(d[:,0],np.log10(d[:,1]),fc_upper[j])
            mr_est.append(a)
            fce_est.append(b)
            fcm_est.append(c)
        ax[1].scatter(fc_upper,fcm_est,s=2)
        ax[1].set_ylim([0.001,4])
        ax[1].set_xlabel('fc_upper')
        ax[1].set_ylabel('fit fc')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        if max(np.log10(fcm_est))-min(np.log10(fcm_est))> 0.5:
            print('{}_{}_{}'.format(data_array[i,0],data_array[i,6],phase))
        fig.tight_layout()
        plt.savefig('{}_{}_{}.png'.format(data_array[i,0],data_array[i,6],phase))
        plt.close()


main(filename=argv[1],phase=argv[2],distance='85')
