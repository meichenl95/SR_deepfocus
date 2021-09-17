#!/home/meichen/anaconda3/bin/python

import numpy as np
import pandas as pd
import os
import glob
from scipy.optimize import curve_fit
from sys import argv
import csv

def func(x,a,b,c):
    return np.log10(a) + 1./2.*np.log10(1 + x**4/b**4) - 1./2.*np.log10(1 + x**4/c**4)

#String to be initialized
phase_distance = "P_rg"
path = "/home/meichen/Research/SR_Attn/pair_events/figures/gcmt/P_rg"


data = pd.read_csv('{}/{}_total.csv'.format(path,phase_distance),skipinitialspace=True)
data_array = np.array(data)

headers = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','{}'.format(phase_distance),'{}_fit'.format(phase_distance),'{}_fc_std'.format(phase_distance),'{}_fc_mean'.format(phase_distance)]
with open('{}/Boatwright/{}_total_Boatwright.csv'.format(path,phase_distance),'w+') as f:
    f_csv = csv.DictWriter(f,headers)
    f_csv.writeheader()
f.close()

for i in np.arange(len(data_array[:,0])):
    filename = glob.glob('{}/bootstrap/{}_{}.all*.sr'.format(path,data_array[i,0],data_array[i,6]))[0]
    d = np.genfromtxt(filename)
    xdata = d[:,0]
    ydata = d[:,1]
    ydata = np.log10(ydata)

    popt,pcov = curve_fit(func,xdata,ydata,bounds=([1,0.025,0.025],[100000,4.,4.]),method='trf',loss='huber',f_scale=0.1)

    #bootstrap
    res = func(xdata,*popt) - ydata
    popt_list = []
    l = len(xdata)
    for count in np.arange(1000):
        random_index = np.random.randint(0,l,size=l)
        new_ydata = ydata + [res[j] for j in random_index]
        try:
            new_popt, pcov = curve_fit(func,xdata,new_ydata,bounds=([1,0.025,0.025],[100000,4.,4.]),method='trf',loss='huber',f_scale=0.1)
            popt_list.append(new_popt)
        except RuntimeError:
            print("Error - curve_fit failed")
    std = np.std(np.array(popt_list)[:,2],ddof=1)
    mean = np.mean(np.array(popt_list)[:,2])
    
    row = [{'masterID':data_array[i,0],'masterTime':data_array[i,1],'mastermag':data_array[i,2],'masterlat':data_array[i,3],'masterlon':data_array[i,4],'masterdep':data_array[i,5],'egfID':data_array[i,6],'egfTime':data_array[i,7],'egfmag':data_array[i,8],'M0':data_array[i,9],'magdif':data_array[i,10],'egflat':data_array[i,11],'egflon':data_array[i,12],'egfdep':data_array[i,13],'{}'.format(phase_distance):data_array[i,14],'{}_fit'.format(phase_distance):popt,'{}_fc_std'.format(phase_distance):std,'{}_fc_mean'.format(phase_distance):mean}]
    with open('{}/Boatwright/{}_total_Boatwright.csv'.format(path,phase_distance),'a') as f:
        f_csv = csv.DictWriter(f,headers)
        f_csv.writerows(row)
    f.close()


