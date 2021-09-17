#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter

def main():
    data = pd.read_csv('pairsfile_rgp_select.csv',skipinitialspace=True)
    data_array = np.array(data)
    jpath = '/home/meichen/work1/SR_Attn/pair_events'
    phase = 'P'
    distance = '85'

    pairs = {}
    masterid = []
    numberid = []
    for i in np.arange(len(data_array[:,0])):
        if data_array[i,0] not in pairs.keys():
            masterid.append(data_array[i,0])
            numberid.append(data_array[i,19])
        pairs.setdefault(data_array[i,0],[]).append(data_array[i,6])

    for key in list(pairs.keys()):
        index=list(data_array[:,0]).index(key)
        os.chdir('{}/master_{}'.format(jpath,key))
        num = 0
        fig = plt.figure(figsize=[4,2])
        ax1 = fig.add_subplot(111)

        for value in list(pairs.get(key)):
            stn_num = glob.glob('egf_{}/{}/gcarc_{}/all*'.format(value,phase,distance))[0].split('.')[7]
            d = np.genfromtxt('{}'.format(glob.glob('egf_{}/{}/gcarc_{}/all*'.format(value,phase,distance))[0]))
            d = d[d[:,0]<2.0]
            indices = [l for l,x in enumerate(data_array[:,0]) if x == key]
            index = list(data_array[l,6] for l in indices).index(value)
            fc = data_array[indices[0]+index,17]
            a = data_array[indices[0]+index,15]
            b = data_array[indices[0]+index,16]

            ax1.loglog(d[:,0],d[:,1],'C{}'.format(num),label='{} stn:{}'.format(value,stn_num),lw=0.5,alpha=0.75)
            ax1.loglog(d[:,0],func_Boatwright(d[:,0],a,b,fc),linestyle='--',color='grey',lw=1)
            ax1.plot(fc,func_Boatwright(fc,a,b,fc),marker='v',markeredgecolor='C{}'.format(num),markerfacecolor='C{}'.format(num),linewidth=2)
            num = num + 1
            num = num % 9
        ax1.set_xlabel('Frequency (Hz)',size=8)
        ax1.set_ylabel('Spectral ratios',size=8)
        ax1.set_xticks([0.025,0.1,1,2])
        ax1.set_xticklabels([0.025,0.1,1,2])
        ax1.yaxis.set_major_locator(mticker.LogLocator(subs=(0.3,1.0,)))
        ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax1.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax1.tick_params(axis='both',which='both',labelsize=6)
        print(key,phase,distance)
        n = masterid.index(key)
        ax1.set_title('# {}'.format(numberid[n]),size=10)
        fig.tight_layout()
        plt.savefig('/home/meichen/Research/SR_Attn/pair_events/figures/master_{}.pdf'.format(numberid[n]))
        plt.close()


def func(x,a,b,c):
    return a * (1 + x**2 / b**2)/(1 + x**2 / c**2)

def func_Boatwright(x,a,b,c):
    return a * (1 + x**4/ b**4)**0.5 / (1+x**4/c**4)**0.5

main()
