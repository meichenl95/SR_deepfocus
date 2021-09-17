#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.signal import detrend
from scipy.interpolate import interp1d
import glob
import os
import obspy

def rsm(x,y):
    x[0] = x[1]
    xl = np.log10(x)
    yl = np.log10(y)
    f = interp1d(xl,yl)
    xn = np.linspace(-1.6,0.9,num=1001)
    yn = f(xn)
    xx = 10**(xn)
    yy = 10**(yn)
    return xx,yy

def smooth(y):
    l = len(y)
    for i in range(2,l-2):
        y[i] = np.mean(y[i-2:i+2])
    return y

def myfft(tr,**kwargs):

##parameters
# tr			the trace to read in
# tpck			t1, t2, ..., t9
# t_b			seconds before tpck of the beginning of the window
# t_a 			seconds after tpck of the end of the window


    tpck = kwargs.get('tpck')
    t_b = kwargs.get('t_b')
    t_a = kwargs.get('t_a')
    
    Fs = tr.stats.sampling_rate
    Ts = tr.stats.delta
    arr_t = tr.stats.sac[tpck]
    p_b = int(np.float(arr_t)/Ts + np.float(t_b)/Ts + (-1)*tr.stats.sac['b']/Ts)
    p_a = int(np.float(arr_t)/Ts + np.float(t_a)/Ts + (-1)*tr.stats.sac['b']/Ts)
    phase_win = tr.data[p_b:p_a]
    phase_win = detrend(phase_win)
    n = len(phase_win)
    nfft = 2**math.ceil(math.log(11*(n+1),2))
    phase_win_add = np.pad(phase_win,(0,10*(n+1)),'constant')
    k = np.arange(nfft)
    T = nfft/Fs
    frq = k/T
    frq = frq[list(range(int(nfft/2)))]
    Y =  np.fft.fft(phase_win_add,n=nfft)/len(phase_win_add)
    Y = abs(Y[list(range(int(nfft/2)))])
    frq,Y = rsm(frq,Y)
    Y = smooth(Y)
    return frq,Y


def snr_plot(**kwargs):

##parameters
# filename		the name of file to read in
# dirname		the directory where read-in files are saved
# datadir		the directory to save data
# phase			'S' or 'P'
# multi			the number of windows
# window_length		the length of each window
# b		seconds before the beginning of the first window before tpck

    dirname = kwargs.get('dirname')
    filename = kwargs.get('filename')
    datadir = kwargs.get('datadir')
    phase = kwargs.get('phase')
    multi = kwargs.get('multi')
    window_length = kwargs.get('window_length')
    b = kwargs.get('b')

    os.chdir('{}'.format(dirname))
    data = pd.read_csv('{}'.format(filename),skipinitialspace=True)
    data_array = np.array(data)
    if phase =='BHZ':
        tpck = 't1'
    elif phase =='BHE' or phase =='BHN':
        tpck = 't2'

    for i in np.arange(len(data_array[:,0])):
        masterid = data_array[i,0]
        egfid = data_array[i,6]
        os.makedirs('{}/master_{}/egf_{}/{}'.format(dirname,masterid,egfid,phase),exist_ok=True)
        os.chdir('{}/master_{}/egf_{}/{}'.format(dirname,masterid,egfid,phase))
        for master_sac in glob.glob('{}/event_{}/waveforms/{}/*.SAC'.format(datadir,masterid,phase)):
            nw,stn,loc,chn = master_sac.split('.')[6:10]
            egf_sac = glob.glob('{}/event_{}/waveforms/{}/*.{}.{}.{}.{}.*.SAC'.format(datadir,egfid,phase,nw,stn,loc,chn))
            if egf_sac != []:
                egf_sac = egf_sac[0]
                tr_master = obspy.read('{}'.format(master_sac))[0]
                tr_egf = obspy.read('{}'.format(egf_sac))[0]

                try:
                    time = tr_master.stats.sac[tpck]
                except:
                    print("{} does not exist of trace {}".format(tpck,tr_master.id))
                    continue
                try:
                    time = tr_egf.stats.sac[tpck]
                except:
                    print("{} does not exist of trace {}".format(tpck,tr_egf.id))
                    continue
                
                if tr_master.stats.delta<0.0625 and tr_egf.stats.delta<0.0625:
                    ## master trace
                    # signal
                    amp_master_sgn = 0
                    for j in np.arange(int(multi)):
                        t_b = np.float(b) + j/2.*np.float(window_length)
                        t_a = np.float(b) + (j/2.+1)*np.float(window_length)
                        frq,master_sgn = myfft(tr_master,tpck=tpck,t_b=t_b,t_a=t_a)
                        amp_master_sgn = amp_master_sgn + master_sgn
                    amp_master_sgn = amp_master_sgn/5.
                    # noise
                    amp_master_noi = 0
                    for j in np.arange(int(multi)):
                        t_b = np.float(b)-(j/2.+1)*np.float(window_length)
                        t_a = np.float(b)-j/2.*np.float(window_length)
                        frq,master_noi = myfft(tr_master,tpck=tpck,t_b=t_b,t_a=t_a)
                        amp_master_noi = amp_master_noi + master_noi
                    amp_master_noi = amp_master_noi/5.
                    ## egf trace
                    # signal
                    amp_egf_sgn = 0
                    for j in np.arange(int(multi)):
                        t_b = np.float(b) + j/2.*np.float(window_length)
                        t_a = np.float(b) + (j/2.+1)*np.float(window_length)
                        frq,egf_sgn = myfft(tr_egf,tpck=tpck,t_b=t_b,t_a=t_a)
                        amp_egf_sgn = amp_egf_sgn + egf_sgn
                    amp_egf_sgn = amp_egf_sgn/5.
                    # noise
                    amp_egf_noi = 0
                    for j in np.arange(int(multi)):
                        t_b = np.float(b) - (j/2.+1)*np.float(window_length)
                        t_a = np.float(b) - j/2.*np.float(window_length)
                        frq,egf_noi = myfft(tr_egf,tpck=tpck,t_b=t_b,t_a=t_a)
                        amp_egf_noi = amp_egf_noi + egf_noi
                    amp_egf_noi = amp_egf_noi/5.
        
                    if np.mean(amp_master_sgn)/np.mean(amp_master_noi) > 2. and np.mean(amp_egf_sgn)/np.mean(amp_egf_noi) > 2.:
                        ##plot figures
                        fig,ax = plt.subplots(2,1,figsize=[6,5])
                        ax[0].plot(frq,amp_master_sgn,color='b',lw=1,alpha=0.75)
                        ax[0].plot(frq,amp_master_noi,color='red',lw=1,alpha=0.75)
                        ax[0].set_xscale('log')
                        ax[0].set_yscale('log')
                        ax[0].set_ylabel('Amp')
                        ax[0].set_title('master_{} {}.{}.{}.{}'.format(masterid,nw,stn,loc,chn))
                        ax[1].plot(frq,amp_egf_sgn,color='b',lw=1,alpha=0.75)
                        ax[1].plot(frq,amp_egf_noi,color='red',lw=1,alpha=0.75)
                        ax[1].set_xscale('log')
                        ax[1].set_yscale('log')
                        ax[1].set_ylabel('Amp')
                        ax[1].set_xlabel('Frequency (Hz)')
                        ax[1].set_title('egf_{}'.format(egfid))
                        fig.tight_layout()
                        plt.savefig('{}_{}.{}.{}.{}.{}.png'.format(masterid,egfid,nw,stn,loc,chn))
                        plt.close()

def main():
    dirname = '/home/meichen/Research/SR_Attn/pair_events/figures/snr'
    datadir = '/home/meichen/work1/SR_Attn/all_events'
    
    for phase in ['BHN','BHE','BHZ']:
        if phase == 'BHE' or phase =='BHN':
            filename = 'pairsfile_rgs_select.csv'
        elif phase == 'BHZ':
            filename = 'pairsfile_rgp_select.csv'
        snr_plot(filename=filename,dirname=dirname,datadir=datadir,phase=phase,multi=5,window_length=40,b=-5)

main()
