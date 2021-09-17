#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x,a,b,c):
    return np.log10(a) + np.log10(1 + x**2/b**2) - np.log10(1 + x**2/c**2)

def mag_test(**kwargs):
    fce_list = kwargs.get('fce_list')

    mr = 30
    fcm = 0.3

    frq = 10**np.linspace(-1.6,0.6,1001)
    amp_all = 0
    for fce in fce_list:
        amp = func(frq,mr,fce,fcm)+np.random.normal(0,0.2,1001)
        amp_all = amp + amp_all
    
    amp_all = 10**(amp_all/len(fce_list))
    fit_mr,fit_fce,fit_fcm = fit(frq,amp_all)

    return fit_fcm

def fit(frq,amp):
    popt,pcov = curve_fit(func,frq,np.log10(amp),bounds=([1,0.,0.],[100000,10.,40.]),method='trf',loss='huber',f_scale=0.1)
    return popt[0],popt[1],popt[2]

def main():
    list1=[1.2,1.3,1.4,1.5,1.6]
    list2=[1.2,1.4,1.6,1.8,2.0]
    list3=[1.2,1.8,2.2,2.6,3.0]
    list4=[1.2,2.0,2.8,3.6,4.4]
    list5=[1.2,2.8,4.4,6.0,7.6]
    list6=[1.2,4.4,7.6,10.8,14.0]
    list7=[7.6,14.0,20.4,26.8,33.2]
    list8=[0.8,0.9,1.0,1.1,1.2]

    fcm_list = []
    for fce_list in list1,list2,list3,list4,list5,list6,list7:
        fcm = mag_test(fce_list=fce_list)
        fcm_list.append(fcm)
    print(fcm_list)

main()
