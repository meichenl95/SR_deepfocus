#!/home/meichen/anaconda3/bin/python
 
def M0r2magdif(M0r):
    import numpy as np
    return 2./3. * np.log10(M0r)


def pairs_select(**kwargs):

##-------------------------------##

# This function select pairs following given criteria.

# Created by Meichen Liu on June 29th, 2019

##-------------------------------##

##parameters
# dirname		the directory to store file and output file
# spec_dirname		the directory to store spectral ratio
# filename		the name of pairs file
# ofile			the name of output file
# stnnum		the minimum number of station records
#			"False" means no criterion on station number
# upper_fc		the upper frequency bound
# lower_fc		the lower frequency bound
# std_range		the maximum standard deviation
#			corner frequency
# mag_dif_dif		the maximum difference between catalog magnitude
#			difference and fitting magnitude difference
# corrcoef		the minimum cross correlation coefficient between
#			spectral ratios and fitting lines
# phase			rs, gs, rgs, rp, gp, rgp


    import os
    import pandas as pd
    import numpy as np
    import glob

    dirname = kwargs.get('dirname')
    spec_dirname = kwargs.get('spec_dirname')
    filename = kwargs.get('filename')
    ofile = kwargs.get('ofile')
    stnnum = kwargs.get('stnnum')
    upper_fc = kwargs.get('upper_fc')
    lower_fc = kwargs.get('lower_fc')
    std_range = kwargs.get('std_range')
    mag_dif_dif = kwargs.get('mag_dif_dif')
    corrcoef = kwargs.get('corrcoef')
    phase = kwargs.get('phase')

    os.chdir('{}'.format(dirname))
    data = pd.read_csv('{}'.format(filename),skipinitialspace=True)
    data_array = np.array(data)

    # determine wave and distance
    if phase == 'rs':
        wave = 'S'
        dist = '30'
    elif phase == 'gs':
        wave = 'S'
        dist = '30_85'
    elif phase == 'rgs':
        wave = 'S'
        dist = '85'
    elif phase == 'rp':
        wave = 'P'
        dist = '30'
    elif phase == 'gp':
        wave = 'P'
        dist = '30_85'
    elif phase == 'rgp':
        wave = 'P'
        dist = '85'

#    # the egf corner freq should be larger than the master corner freq
#    data_array = data_array[data_array[:,17]<data_array[:,16]]

    # number of station records
    if stnnum != "False":
        data_array = data_array[data_array[:,14]>stnnum]

#    # large purturbation & cross correlation
#    l = []
#    for i in np.arange(len(data_array[:,0])):
#        masterid = data_array[i,0]
#        egfid = data_array[i,6]
#        M0r = data_array[i,15]
#        fcegf = data_array[i,16]
#        fcmaster = data_array[i,17]
#        spec_r = np.genfromtxt('{}'.format(glob.glob('{}/master_{}/egf_{}/{}/gcarc_{}/all.*'.format(spec_dirname,masterid,egfid,wave,dist))[0]))
#        syn_r = func(np.array(spec_r[:,0]),M0r,fcegf,fcmaster)
#        if np.log10(np.max(spec_r[:,1])/np.max(syn_r))/np.log10(np.max(syn_r)/np.min(syn_r)) > 1:
#            l.append(i)
#        elif np.log10(np.min(syn_r)/np.min(spec_r[:,1]))/np.log10(np.max(syn_r)/np.min(syn_r)) > 1:
#            l.append(i)
#        elif np.mean(abs(np.log10(spec_r[:,1])-np.log10(syn_r)))/np.log10(np.max(syn_r)/np.min(syn_r)) > 0.25:
#            l.append(i)
#        elif np.corrcoef(np.log10(spec_r[:,1]),np.log10(syn_r))[0,1] < float(corrcoef):
#            l.append(i)
#    data_array = np.delete(data_array,np.array(l),axis=0)

    # frequency boundary
    data_array = data_array[data_array[:,17]<upper_fc]
    data_array = data_array[data_array[:,17]>lower_fc]

    # the range of standard deviation
    data_array = data_array[data_array[:,18]<std_range]

    # magnitude difference
    l = []
    for i in np.arange(len(data_array[:,0])):
        magdif = M0r2magdif(data_array[i,15])
        if abs(magdif - data_array[i,10]) > mag_dif_dif:
            l.append(i)
    data_array = np.delete(data_array,np.array(l),axis=0)
    
    # write a new csv
    df = pd.DataFrame()
    for i in np.arange(len(data.columns)):
        df['{}'.format(data.columns[i])] = data_array[:,i]
    df.to_csv('{}'.format(ofile),index=False)

def func(x,a,b,c):
    return a * (1+x**2/b**2) / (1+x**2/c**2)



def main():
    import numpy as np

    dirname = '/home/meichen/Research/SR_Attn/pair_events/search_pairs'
    spec_dirname = '/home/meichen/work1/SR_Attn/pair_events'
    for phase in ['rgs','rgp']:
        pairs_select(dirname=dirname,spec_dirname=spec_dirname,filename='pairsfile_neic_{}.csv'.format(phase),ofile='pairsfile_{}_select.csv'.format(phase),phase=phase,stnnum=1,upper_fc=30,lower_fc=0.0,std_range=5,corrcoef=0.0,mag_dif_dif=5)

main()
