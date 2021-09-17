#!/home/meichen/anaconda3/bin/python

def add_info(**kwargs):
##**************************************##

# This function add event info to SAC files, including event latitude, event
# longitude, event depth, event magnitude, event original time.

# Created by Meichen Liu on June 10th, 2019 based on sac-manual-v3.6
##**************************************##

##parameters
# filename      the name of the SAC file
# dirname       the directory where SAC files are saved
# origin        the original time of the event
# evlo          the longitude of the event
# evla          the latitude of the event
# evdp		the depth of the event
# mag           the magnitude of the event

    import os
    import sys
    import datetime
    import subprocess
    import glob
    import numpy as np

    filename = kwargs.get('filename')
    dirname = kwargs.get('dirname')
    origin = kwargs.get('origin')
    evlo = kwargs.get('evlo')
    evla = kwargs.get('evla')
    evdp = kwargs.get('evdp')
    mag = kwargs.get('mag')

    os.putenv("SAC_DISPLAY_COPYRIGHT","O")
    os.chdir('{}'.format(dirname))

    o = datetime.datetime.strptime(origin,'%Y-%m-%dT%H:%M:%S')
    # calculate which day in a year is the occurence date
    jday = o.strftime("%j")

    p = subprocess.Popen(['sac'],stdin=subprocess.PIPE)
    s = "wild echo off \n"

    filelist = glob.glob('{}'.format(filename))
    if len(filelist)>20:
        for i in np.arange(round(len(filelist)/20)):
            s += "r %s \n" % filelist[i*20:i*20+20]
            s += "synchronize \n"
            s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
            s += "ch allt (0 - &1,o&) iztype IO \n"
            s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
            s += "wh \n"
        s += "r %s \n" % filelist[i*20+20::]
        s += "synchronize \n"
        s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
        s += "ch allt (0 - &1,o&) iztype IO \n"
        s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
        s += "wh \n"
    else:
        s += "r %s \n" % filename
        s += "synchronize \n"
        s += "ch o gmt %s %s %s %s %s \n" % (o.year, jday, o.hour, o.minute, o.second)
        s += "ch allt (0 - &1,o&) iztype IO \n"
        s += "ch evlo %s evla %s evdp %s mag %s \n" % (evlo, evla, evdp, mag)
        s += "wh \n"
    s += "q \n"
    p.communicate(s.encode())


def arrival_time(**kwargs):                                                        
                                                                                   
##**************************************##                                         
                                                                                   
# This function save the predicted arrival time of desired phase.                  
                                                                                   
# Created by Meichen Liu on June 10th, 2019                                        
                                                                                   
##**************************************##                                         
                                                                                   
##parameters                                                                       
# filename      the name of sac file                                               
# dirname       the directory seismograms are saved                                
# phase         name of the phase                                                  
# model         the model to calculate travel time.                                
#               "prem", "iasp91","ak135"                                           
# Tn            the number of Tn to store travel time                              
                                                                                   
    import os                                                                      
    import glob                                                                    
    import subprocess                                                              
    import numpy as np                                                             
                                                                                   
    filename = kwargs.get('filename')                                              
    dirname = kwargs.get('dirname')                                                
    phase = kwargs.get('phase')                                                    
    model = kwargs.get('model')                                                    
    Tn = kwargs.get('Tn')                                                          
                                                                                   
    os.chdir('{}'.format(dirname))                                                 
                                                                                   
    for sacfile in glob.glob('{}'.format(filename)):                               
        subprocess.call(['taup_setsac','-mod',model,'-ph','{}-{}'.format(phase,Tn),'-evdpkm','{}'.format(sacfile)])

def table_mark(**kwargs):                                                       
                                                                                
##**************************************##                                      
                                                                                
# This function mark arrival time based on self-made timetable.                 
                                                                                
##**************************************##                                      
                                                                                
##parameters                                                                    
# dirname       The directory where sacfiles are saved                          
# filename      the name of sac files                                           
# ofile         the appendix of output files. False means no appendix           
# tablef        the timetable file                                              
# Tn            to save arrival time                                            
                                                                                
                                                                                
    import numpy as np                                                          
    import subprocess                                                           
    import os                                                                   
    import glob                                                                 
                                                                                
    dirname = kwargs.get('dirname')                                             
    filename = kwargs.get('filename')                                           
    ofile = kwargs.get('ofile')                                                 
    tablef = kwargs.get('tablef')                                               
    Tn = kwargs.get('Tn')                                                       
                                                                                
    os.chdir('{}'.format(dirname))                                              
    os.putenv('SAC_DISPLAY_COPYRIGHT','O')                                      
                                                                                
    p = subprocess.Popen(['sac'],stdin=subprocess.PIPE)                         
    s = "wild echo off\n"                                                       
                                                                                
    # read time table: distance(deg) depth(km) phase traveltime ray_param(s/deg) takeoff(deg) incident(deg)
    table = np.genfromtxt(tablef)                                               
    for sacfile in glob.glob('{}'.format(filename)):                            
        b, e, o, evdp, gcarc = subprocess.check_output(['saclst','b','e','o','evdp','gcarc','f',sacfile]).decode().split()[1::]
        gcarc = round(float(gcarc))                                             
        evdp_lower = np.floor(float(evdp)/10)*10                                   
        evdp_upper = evdp_lower + 10                                            
        gcarc_index = np.where(table[:,0] == gcarc)[0]                          
        evdp_lower_index = np.where(table[:,1] == evdp_lower)[0]                
        lower_index = [x for x in gcarc_index if x in evdp_lower_index]         
        tt_lower = table[lower_index,2]                                         
        evdp_upper_index = np.where(table[:,1] == evdp_upper)[0]                
        upper_index = [x for x in gcarc_index if x in evdp_upper_index]         
        tt_upper = table[upper_index,2]                                         
        if tt_lower != 999999 and tt_upper != 999999:
            tt = (float(evdp) - evdp_lower)/10 * (float(tt_upper) -float(tt_lower)) + float(tt_lower)
            if tt>float(b) and tt<float(e):
                s += "r %s\n" % sacfile
                s += "ch %s %f\n" % (Tn, tt)                                        
                s += "wh\n"                                                         
                if ofile != "False":                                                
                    s += "w append %s\n" % ofile                                    
                                                                                
    s += "q \n"                                                                 
    p.communicate(s.encode())

def rsm(x,y):
    import numpy as np
    from scipy.interpolate import interp1d
    x[0] = x[1]
    xl = np.log10(x)
    yl = np.log10(y)
    f = interp1d(xl,yl)
    xn = np.linspace(-1.6,0.6,num=1001)
    yn = f(xn)
    xx = 10**(xn)
    yy = 10**(yn)
    return xx,yy

def smooth(y):
    import numpy as np
    l = len(y)
    for i in range(2,l-2):
        y[i] = np.mean(y[i-2:i+2])
    return y

def myfft(**kwargs):

##**************************************##

# This function do fast fourier transformation to trace within a certain time
# window.

##**************************************##

##parameters
# arr_t			the reference time marked in sac header
# t_b			seconds before arr_t
# t_a			seconds after arr_t
# tr			trace in obspy

    import obspy
    from sys import argv
    from scipy.signal import detrend
    from scipy.signal import tukey
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    tr = kwargs.get('tr')
    arr_t = kwargs.get('arr_t')
    t_b = kwargs.get('t_b')
    t_a = kwargs.get('t_a')
    Fs = tr.stats.sampling_rate
    Ts = tr.stats.delta
    start = tr.stats.sac['b']
    arr_t = tr.stats.sac[arr_t]
    p_b = int(float(arr_t)/Ts + float(t_b)/Ts - float(start)/Ts)
    p_a = int(float(arr_t)/Ts + float(t_a)/Ts - float(start)/Ts) 
    phase_win = tr.data[p_b:p_a] 
    phase_win = detrend(phase_win) 
    n = len(phase_win) 
    phase_win = tukey(n,0.1) * phase_win 
    nfft = 2**math.ceil(math.log(11*(n+1),2)) 
    phase_win_add = np.pad(phase_win,(0,10*(n+1)),'constant')
    k = np.arange(nfft) 
    T = nfft/Fs
    frq = k/T 
    frq = frq[list(range(int(nfft/2)))]
    Y = np.fft.fft(phase_win_add,n=nfft)/len(phase_win_add) 
    Y = abs(Y[list(range(int(nfft/2)))])
    frq,Y = rsm(frq,Y)
    Y = smooth(Y)
    return frq,Y      


def mk_path(path):                                                              

    import os
    import subprocess

    isExist = os.path.exists(path)                                              

    if isExist:                                                                 
        subprocess.call(['rm -r {}'.format(path)],shell=True)                   
    os.makedirs(path) 


def my_snr(**kwargs):

##**************************************##

# This function delete those low snr traces using multi-moving window.

##**************************************##

##parameters
# dirname		the directory where traces are stored
# b			seconds before the arrival time
# window_length		length of each window
# multi			the number of window

    import os
    import obspy
    import subprocess
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    dirname = kwargs.get('dirname')
    b = kwargs.get('b')
    window_length = kwargs.get('window_length')
    multi = kwargs.get('multi')

    for phase in ['BHE','BHN']:
        os.chdir('{}'.format(dirname))
        mk_path('{}/{}'.format(dirname,phase))
        subprocess.call(['cp *.{}.*.SAC {}/'.format(phase,phase)],shell=True)
        mk_path('{}/{}/gcarc_30'.format(dirname,phase)) 
        mk_path('{}/{}/gcarc_30_85'.format(dirname,phase)) 
        filename = glob.glob('{}/{}/*.SAC'.format(dirname,phase))
        if phase == 'BHZ':
            arr_t = 't1'
        elif phase == 'BHE' or phase == 'BHN':
            arr_t = 't1'
        for i in np.arange(len(filename)):
            amp_all = 0
            noise_amp_all = 0
            tr = obspy.read(filename[i])[0]

            try:
                time = tr.stats.sac[arr_t]
            except:
                print ("{} does not exist of trace {}".format(arr_t,tr.id))
                continue

            if tr.stats.sac[arr_t] > 0 and tr.stats.delta <= 0.125:
                for j in np.arange(int(multi)):
                    t_b = float(b) + j/2.*float(window_length)
                    t_a = float(b) + (j/2.+1)*float(window_length)
                    flag = 0
                    try:
                        frq,amp = myfft(tr=tr,arr_t=arr_t,t_b=t_b,t_a=t_a) 
                    except:
                        print("signal fft failed of trace {}".format(tr.id))
                        flag=1
                    t_b = float(b)-(j/2.+1)*float(window_length) 
                    t_a = float(b)-j/2.*float(window_length) 
                    try:
                        noise_frq,noise_amp = myfft(tr=tr,arr_t=arr_t,t_b=t_b,t_a=t_a) 
                    except:
                        print("noise fft failed of trace {}".format(tr.id))
                        flag=1

                    amp_all = amp + amp_all
                    noise_amp_all = noise_amp_all + noise_amp 

                if flag == 1:
                    continue
                amp_all = amp_all/float(multi)
                noise_amp_all = noise_amp_all/float(multi)
                phase_amp = np.vstack((frq,amp_all)).T 
                noise_amp = np.vstack((noise_frq,noise_amp_all)).T 

                # 0.025-0.1 Hz
                phase_band1 = phase_amp[phase_amp[:,0]>0.025] 
                phase_band1 = phase_band1[phase_band1[:,0]<0.1] 
                phase_band1_mean = np.mean(phase_band1[:,1])
                noise_band1 = noise_amp[noise_amp[:,0]>0.025] 
                noise_band1 = noise_band1[noise_band1[:,0]<0.1] 
                noise_band1_mean = np.mean(noise_band1[:,1])    
                # 0.1-0.4 Hz
                phase_band2 = phase_amp[phase_amp[:,0]>0.1]
                phase_band2 = phase_band2[phase_band2[:,0]<0.4] 
                phase_band2_mean = np.mean(phase_band2[:,1])
                noise_band2 = noise_amp[noise_amp[:,0]>0.1]
                noise_band2 = noise_band2[noise_band2[:,0]<0.4] 
                noise_band2_mean = np.mean(noise_band2[:,1])    
                # 0.4-0.9 Hz
                phase_band3 = phase_amp[phase_amp[:,0]>0.4]
                phase_band3 = phase_band3[phase_band3[:,0]<0.9] 
                phase_band3_mean = np.mean(phase_band3[:,1])
                noise_band3 = noise_amp[noise_amp[:,0]>0.4]
                noise_band3 = noise_band3[noise_band3[:,0]<0.9] 
                noise_band3_mean = np.mean(noise_band3[:,1])
                # 0.9-2 Hz
                phase_band4 = phase_amp[phase_amp[:,0]>0.9]
                phase_band4 = phase_band4[phase_band4[:,0]<2.0] 
                phase_band4_mean = np.mean(phase_band4[:,1])
                noise_band4 = noise_amp[noise_amp[:,0]>0.9]
                noise_band4 = noise_band4[noise_band4[:,0]<2.0] 
                noise_band4_mean = np.mean(noise_band4[:,1])
    
                flag = 0
                if noise_band1_mean > 0. and noise_band2_mean >0. and noise_band3_mean >0. and noise_band4_mean >0.: 
                    band1_snr = phase_band1_mean/noise_band1_mean
                    band2_snr = phase_band2_mean/noise_band2_mean
                    band3_snr = phase_band3_mean/noise_band3_mean
                    band4_snr = phase_band4_mean/noise_band4_mean
                    if band1_snr >2. and band2_snr>2. and band3_snr>2. and band4_snr>2.:
                        flag = 1
    
                if flag == 1:     
                    if tr.stats.sac['gcarc'] < 30: 
                        try: 
                            subprocess.call(['cp {} {}/{}/gcarc_30'.format(filename[i],dirname,phase)],shell=True)
                        except: 
                            print('Failed to snr {}'.format(filename[i])) 
                    elif tr.stats.sac['gcarc'] >30 and tr.stats.sac['gcarc'] < 85:
                        try: 
                            subprocess.call(['cp {} {}/{}/gcarc_30_85'.format(filename[i],dirname,phase)],shell=True)
                        except: 
                            print('Failed to snr {}'.format(filename[i])) 
                    # plot figures
                    fig,ax = plt.subplots(1,1,figsize=[5,3])
                    ax.plot(phase_amp[:,0],phase_amp[:,1],color='b',lw=1,alpha=0.75)
                    ax.plot(noise_amp[:,0],noise_amp[:,1],color='orange',lw=1,alpha=0.75)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlabel('Frequency (Hz)')
                    ax.set_ylabel('Amp')
                    ax.set_title('{}'.format(filename[i]))
                    plt.savefig('{}.png'.format(filename[i]))
                    plt.close()
                    
                                                                                

def main():
    
    import numpy as np
    import pandas as pd

    file_path = '/home/meichen/work1/SR_Attn/all_events'
    uni_id = pd.read_csv('uni_id.txt',skipinitialspace=True,header=None)
    uni_id = np.array(uni_id)

    for eventid in uni_id[415:416,0]:
        index = list(uni_id[:,0]).index(eventid)
        evtime = uni_id[index,1]
        evla = uni_id[index,2]
        evlo = uni_id[index,3]
        evdp = uni_id[index,4]
        evmag = uni_id[index,5]
        print(eventid,evtime,evla,evlo,evdp)
        # add evla evlo evdp mag
        add_info(dirname='{}/event_{}/waveforms'.format(file_path,eventid),filename='*.SAC',origin=evtime,evla=evla,evlo=evlo,evdp=evdp,mag=evmag)
        # add arrival time T1 for P wave, and T2 for S wave
        table_mark(dirname='{}/event_{}/waveforms'.format(file_path,eventid),filename='*.SAC',tablef='/home/meichen/Utils/TauP_calculations/TravelTimeTables/tt.prem.S',Tn='T2',ofile='False')
        table_mark(dirname='{}/event_{}/waveforms'.format(file_path,eventid),filename='*.SAC',tablef='/home/meichen/Utils/TauP_calculations/TravelTimeTables/tt.prem.P',Tn='T1',ofile='False')
#        arrival_time(filename='*.SAC',dirname='{}/event_{}/waveforms'.format(file_path,eventid),phase='S',model='prem',Tn='2')
#        arrival_time(filename='*.SAC',dirname='{}/event_{}/waveforms'.format(file_path,eventid),phase='P',model='prem',Tn='1')
        # do signal-noise-ratio
        my_snr(dirname='{}/event_{}/waveforms'.format(file_path,eventid),b=-5,window_length=40,multi=5)

main()

