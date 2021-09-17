#!/home/meichen/anaconda3/bin/python
import numpy as np
import pandas as pd
import os
import obspy
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader
import glob
import shutil
import subprocess
from scipy.optimize import curve_fit
from sys import argv
import csv
import sys

def main():

    saveout = sys.stdout
    saveerr = sys.stderr
    f = open('stdout.log','w')
    sys.stderr = f
    sys.stdout = f

    events = pd.read_csv('events_mag5.5_neic.txt',sep='|',skipinitialspace=True)
    events_list = events
    events = np.array(events)
    main_path = "/home/meichen/Research/SR_Attn/pair_events/search_pairs"
    jpath="/home/meichen/work1/SR_Attn"
    window_length = 40
    
    lat = []
    lon = []
    radius = []
    for i in np.arange(events.shape[0]):
        lat.append(np.float(events[i,13])*(np.pi)/180.)
        lon.append(np.float(events[i,14])*(np.pi)/180.)
        radius.append(6371-np.float(events[i,15]))
    num=0
    pairs = {}
    d = []
    
    for i in np.arange(events.shape[0]):
        if np.float(events[i,16])>=8:
            for j in np.arange(events.shape[0]):
                if np.float(events[j,16])>=5.5:
                    if (np.float(events[i,16])-np.float(events[j,16]))>=0.5:
                        arg=np.cos(lat[i])*np.cos(lat[j])*np.cos(lon[i]-lon[j])+np.sin(lat[i])*np.sin(lat[j])
                        dist=np.sqrt(radius[i]**2+radius[j]**2-2*radius[i]*radius[j]*arg)
                        if dist<=500:
                            d.append(events[i,0])
                            d.append(events[j,0])
                            pairs.setdefault(events[i,0],[]).append(events[j,0])
        if np.float(events[i,16])>=7 and np.float(events[i,16])<8:
            for j in np.arange(events.shape[0]):
                if np.float(events[j,16])>=5.5:
                    if (np.float(events[i,16])-np.float(events[j,16]))>=0.5:
                        arg=np.cos(lat[i])*np.cos(lat[j])*np.cos(lon[i]-lon[j])+np.sin(lat[i])*np.sin(lat[j])
                        dist=np.sqrt(radius[i]**2+radius[j]**2-2*radius[i]*radius[j]*arg)
                        if dist<=300:
                            d.append(events[i,0])
                            d.append(events[j,0])
                            pairs.setdefault(events[i,0],[]).append(events[j,0])
        if np.float(events[i,16])>=6 and np.float(events[i,16])<7:
            for j in np.arange(events.shape[0]):
                if np.float(events[j,16])>=5.5:
                    if (np.float(events[i,16])-np.float(events[j,16]))>=0.5:
                        arg=np.cos(lat[i])*np.cos(lat[j])*np.cos(lon[i]-lon[j])+np.sin(lat[i])*np.sin(lat[j])
                        dist=np.sqrt(radius[i]**2+radius[j]**2-2*radius[i]*radius[j]*arg)
                        if dist<=100:
                            d.append(events[i,0])
                            d.append(events[j,0])
                            pairs.setdefault(events[i,0],[]).append(events[j,0])
    
    uni_d = np.unique(np.array(d))
#save text file
    with open("uni_id.txt",'w') as f:
        for key in uni_d:
            index = list(events[:,0]).index(key)
            f.write("%s,%s,%s,%s,%s,%s\n" % (events_list['#EventID '][index], events_list['Time '][index], events_list['Latitude '][index], events_list['Longitude '][index], events_list['Depth/km '][index], events_list['Magnitude '][index]))
    f.close()


##make directory; download data; mseedfile to sacfile
#    os.chdir('{}/all_events'.format(jpath))
#    for key in uni_d:
#        isExist=os.path.exists('event_{}'.format(key))
#        if isExist:
#            continue
#        else:
#            mk_path('event_{}'.format(key))
#            index=list(events[:,0]).index(key)
#            download_data(events[index,:],"event_{}".format(key))
#            mseed2sac('event_{}'.format(key))

##process data now for each egf     
#    header_rs = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','rs','rs_M0r','rs_fcegf','rs_fcmaster','rs_fc_std']
#    header_gs = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','gs','gs_M0r','gs_fcegf','gs_fcmaster','gs_fc_std']
    header_rgs = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','rgs','rgs_M0r','rgs_fcegf','rgs_fcmaster','rgs_fc_std']
#    header_rp = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','rp','rp_M0r','rp_fcegf','rp_fcmaster','rp_fc_std']
#    header_gp = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','gp','gp_M0r','gp_fcegf','gp_fcmaster','gp_fc_std']
    header_rgp = ['masterID','masterTime','mastermag','masterlat','masterlon','masterdep','egfID','egfTime','egfmag','M0','magdif','egflat','egflon','egfdep','rgp','rgp_M0r','rgp_fcegf','rgp_fcmaster','rgp_fc_std']
#    with open('{}/pairsfile_neic_rs.csv'.format(main_path),'w+') as f_rs:
#        f_csv_rs = csv.DictWriter(f_rs,header_rs)
#        f_csv_rs.writeheader()
#    with open('{}/pairsfile_neic_gs.csv'.format(main_path),'w+') as f_gs:
#        f_csv_gs = csv.DictWriter(f_gs,header_gs)
#        f_csv_gs.writeheader()
    with open('{}/pairsfile_neic_rgs_mtspec.csv'.format(main_path),'w+') as f_rgs:
        f_csv_rgs = csv.DictWriter(f_rgs,header_rgs)
        f_csv_rgs.writeheader()
#    with open('{}/pairsfile_neic_rp.csv'.format(main_path),'w+') as f_rp:
#        f_csv_rp = csv.DictWriter(f_rp,header_rp)
#        f_csv_rp.writeheader()
#    with open('{}/pairsfile_neic_gp.csv'.format(main_path),'w+') as f_gp:
#        f_csv_gp = csv.DictWriter(f_gp,header_gp)
#        f_csv_gp.writeheader()
    with open('{}/pairsfile_neic_rgp_mtspec.csv'.format(main_path),'w+') as f_rgp:
        f_csv_rgp = csv.DictWriter(f_rgp,header_rgp)
        f_csv_rgp.writeheader()

    row = []                      
    for key in list(pairs.keys()):
        print("master {}".format(key))
#        mk_path('{}/pair_events/master_{}'.format(jpath,key))
        index = list(events[:,0]).index(key)
        masterTime = events[index,1]                               
        mastermag = "{}".format(events[index,16])
        masterlat = events[index,13]
        masterlon = events[index,14]
        masterdep = events[index,15]
        M0 = np.float(events[index,18])*1e-7
        flag = 0
        for value in list(pairs.get(key)):
            print("egf {}".format(value))
#            mk_path('{}/pair_events/master_{}/egf_{}'.format(jpath,key,value))
            index = list(events[:,0]).index(value)
            egfTime = events[index,1]                              
            egfmag = "{}".format(events[index,16]) 
            egflat = events[index,13]
            egflon = events[index,14]
            egfdep = events[index,15]
            magdif = np.float(mastermag) - np.float(egfmag)

            each_process(jpath,key,value,'P','30')
            each_process(jpath,key,value,'P','30_85')
            all_85(jpath,key,value,'P')
            each_process(jpath,key,value,'S','30')
            each_process(jpath,key,value,'S','30_85')
            all_85(jpath,key,value,'S')
#            rs_M0r,rs_fcegf,rs_fcmaster,rs_fc_std,rs_stnnm = fit(main_path=jpath,master=key,egf=value,phase="S",dist_range="30",window_length=window_length)
#            gs_M0r,gs_fcegf,gs_fcmaster,gs_fc_std,gs_stnnm = fit(main_path=jpath,master=key,egf=value,phase="S",dist_range="30_85",window_length=window_length)
            rgs_M0r,rgs_fcegf,rgs_fcmaster,rgs_fc_std,rgs_stnnm = fit(main_path=jpath,master=key,egf=value,phase="S",dist_range="85",window_length=window_length)
#            rp_M0r,rp_fcegf,rp_fcmaster,rp_fc_std,rp_stnnm = fit(main_path=jpath,master=key,egf=value,phase="P",dist_range="30",window_length=window_length)
#            gp_M0r,gp_fcegf,gp_fcmaster,gp_fc_std,gp_stnnm = fit(main_path=jpath,master=key,egf=value,phase="P",dist_range="30_85",window_length=window_length)
            rgp_M0r,rgp_fcegf,rgp_fcmaster,rgp_fc_std,rgp_stnnm = fit(main_path=jpath,master=key,egf=value,phase="P",dist_range="85",window_length=window_length)

#            row_rs = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'rs':rs_stnnm,'rs_M0r':rs_M0r,'rs_fcegf':rs_fcegf,'rs_fcmaster':rs_fcmaster,'rs_fc_std':rs_fc_std}]
#            row_gs = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'gs':gs_stnnm,'gs_M0r':gs_M0r,'gs_fcegf':gs_fcegf,'gs_fcmaster':gs_fcmaster,'gs_fc_std':gs_fc_std}]
            row_rgs = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'rgs':rgs_stnnm,'rgs_M0r':rgs_M0r,'rgs_fcegf':rgs_fcegf,'rgs_fcmaster':rgs_fcmaster,'rgs_fc_std':rgs_fc_std}]
#            row_rp = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'rp':rp_stnnm,'rp_M0r':rp_M0r,'rp_fcegf':rp_fcegf,'rp_fcmaster':rp_fcmaster,'rp_fc_std':rp_fc_std}]
#            row_gp = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'gp':gp_stnnm,'gp_M0r':gp_M0r,'gp_fcegf':gp_fcegf,'gp_fcmaster':gp_fcmaster,'gp_fc_std':gp_fc_std}]
            row_rgp = [{'masterID':key,'masterTime':masterTime,'mastermag':mastermag,'masterlat':masterlat,'masterlon':masterlon,'masterdep':masterdep,'egfID':value,'egfTime':egfTime,'egfmag':egfmag,'M0':M0,'magdif':magdif,'egflat':egflat,'egflon':egflon,'egfdep':egfdep,'rgp':rgp_stnnm,'rgp_M0r':rgp_M0r,'rgp_fcegf':rgp_fcegf,'rgp_fcmaster':rgp_fcmaster,'rgp_fc_std':rgp_fc_std}]
#            with open('{}/pairsfile_neic_rs.csv'.format(main_path),'a') as f_rs:
#                f_csv_rs = csv.DictWriter(f_rs,header_rs)
#                f_csv_rs.writerows(row_rs)
#            with open('{}/pairsfile_neic_gs.csv'.format(main_path),'a') as f_gs:
#                f_csv_gs = csv.DictWriter(f_gs,header_gs)
#                f_csv_gs.writerows(row_gs)
            with open('{}/pairsfile_neic_rgs_mtspec.csv'.format(main_path),'a') as f_rgs:
                f_csv_rgs = csv.DictWriter(f_rgs,header_rgs)
                f_csv_rgs.writerows(row_rgs)
#            with open('{}/pairsfile_neic_rp.csv'.format(main_path),'a') as f_rp:
#                f_csv_rp = csv.DictWriter(f_rp,header_rp)
#                f_csv_rp.writerows(row_rp)
#            with open('{}/pairsfile_neic_gp.csv'.format(main_path),'a') as f_gp:
#                f_csv_gp = csv.DictWriter(f_gp,header_gp)
#                f_csv_gp.writerows(row_gp)
            with open('{}/pairsfile_neic_rgp_mtspec.csv'.format(main_path),'a') as f_rgp:
                f_csv_rgp = csv.DictWriter(f_rgp,header_rgp)
                f_csv_rgp.writerows(row_rgp)
            
    sys.stdout = saveout
    sys.stderr = saveerr
    f.close()

def func(x,a,b,c):
    return np.log10(a) + np.log10(1 + x**2 / b**2) - np.log10(1 + x**2 / c**2)


def func_Boatwright(x,a,b,c):
    return np.log10(a) + 1./2. * np.log10(1 + x**4 / b**4) - 1./2. * np.log10(1 + x**4 / c**4)

def fit(**kwargs):
    main_path = kwargs.get('main_path')
    master = kwargs.get('master')
    egf = kwargs.get('egf')
    phase = kwargs.get('phase')                              
    dist_range = kwargs.get('dist_range')                    
    window_length = kwargs.get('window_length')              
    filename = glob.glob('{}/pair_events/master_{}/egf_{}/{}/gcarc_{}/all*.stn*.Np.sr'.format(main_path,master,egf,phase,dist_range))
    print(filename[0].split('/')[-1])
    stn_num = filename[0].split('/')[-1].split(".")[7]
    if int(stn_num) != 0:                                    
        # read in data and cut needed                        
        data = np.genfromtxt(filename[0])
#        data = data[data[:,0]>=float(1./float(window_length))]
        data = data[data[:,0]<=2.]
        xdata = data[:,0]                                    
        ydata = data[:,1]                                    
        ydata = np.log10(ydata)                              
                                                             
        # find best fit model
        try:
            popt, pcov = curve_fit(func, xdata, ydata, bounds=([1,0.0,0.0],[100000,100.,100.]),method='trf',loss='huber',f_scale=0.1)
                                                             
            ## uncertainty analysis by bootstrapping            
            # calculate residuals                                
            res = ydata-func(xdata,*popt)                        
            popt_list = []                                       
            # length of xdata                                    
            l = len(xdata)                                       
            # bootstrap                                          
            for i in np.arange(50):                            
                random_index = np.random.randint(0,l,size=l)     
                new_ydata = func(xdata,*popt) + [res[j] for j in random_index] 
                try:
                    new_popt, new_pcov = curve_fit(func, xdata,new_ydata, bounds=([1,0.0,0.0],[100000,100.,100.]),method='trf',loss='huber',f_scale=0.1)
                    popt_list.append(new_popt)
                except RuntimeError:
                    print("Error - curve_fit failed")
            std = np.std(np.log10(np.array(popt_list)[:,2]),ddof=1)
#            fcmaster = 10**(np.mean(np.log10(np.array(popt_list)[:,2])))
#            M0r = 10**(np.mean(np.log10(np.array(popt_list)[:,0])))
#            fcegf = 10**(np.mean(np.log10(np.array(popt_list)[:,1])))
                                                                 
            return popt[0],popt[1],popt[2],std,stn_num
        except:
            print("Error - Curve fit failed")
            return 0,0,0,0,0
    else:                                                    
        return 0,0,0,0,0



def all_85(path,key,value,phase):
    mk_path('{}/pair_events/master_{}/egf_{}/{}/gcarc_85'.format(path,key,value,phase))
    os.chdir('{}/pair_events/master_{}/egf_{}/{}/gcarc_85'.format(path,key,value,phase))
    window_begin=-5
    window_length=40
    window_multi=5
    subprocess.call(['cp {}/pair_events/master_{}/egf_{}/{}/gcarc_30/*.Np*.master {}/pair_events/master_{}/egf_{}/{}/gcarc_85/'.format(path,key,value,phase,path,key,value,phase)],shell=True)
    subprocess.call(['cp {}/pair_events/master_{}/egf_{}/{}/gcarc_30_85/*.Np*.master {}/pair_events/master_{}/egf_{}/{}/gcarc_85/'.format(path,key,value,phase,path,key,value,phase)],shell=True)
    subprocess.call(['cp {}/pair_events/master_{}/egf_{}/{}/gcarc_30/*.Np*.egf {}/pair_events/master_{}/egf_{}/{}/gcarc_85/'.format(path,key,value,phase,path,key,value,phase)],shell=True)
    subprocess.call(['cp {}/pair_events/master_{}/egf_{}/{}/gcarc_30_85/*.Np*.egf {}/pair_events/master_{}/egf_{}/{}/gcarc_85/'.format(path,key,value,phase,path,key,value,phase)],shell=True)
    subprocess.call(['bash','/home/meichen/bin/sr_calc.sh','{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'{}'.format(phase),'85'])
    

def each_process(path,key,value,phase,gcarc):
    mk_path('{}/pair_events/master_{}/egf_{}/{}/gcarc_{}'.format(path,key,value,phase,gcarc))
    os.chdir('{}/pair_events/master_{}/egf_{}/{}/gcarc_{}'.format(path,key,value,phase,gcarc))
    window_begin=-5
    window_length=40
    window_multi=5
    if phase == 'P':
        subprocess.call(['bash','/home/meichen/bin/stn_sel.sh','{}/all_events/event_{}/waveforms/BHZ/gcarc_{}'.format(path,key,gcarc),'{}/all_events/event_{}/waveforms/BHZ/gcarc_{}'.format(path,value,gcarc),'{}/pair_events/master_{}/egf_{}/P/gcarc_{}'.format(path,key,value,gcarc)])
        subprocess.call(['python','/home/meichen/bin/seismoscript_mtspec.py',"*.master","t1",'{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'master'])
        subprocess.call(['python','/home/meichen/bin/seismoscript_mtspec.py',"*.egf","t1",'{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'egf'])
    elif phase == 'S':
        subprocess.call(['bash','/home/meichen/bin/stn_sel.sh','{}/all_events/event_{}/waveforms/BHN/gcarc_{}'.format(path,key,gcarc),'{}/all_events/event_{}/waveforms/BHN/gcarc_{}'.format(path,value,gcarc),'{}/pair_events/master_{}/egf_{}/S/gcarc_{}'.format(path,key,value,gcarc)])
        subprocess.call(['bash','/home/meichen/bin/stn_sel.sh','{}/all_events/event_{}/waveforms/BHE/gcarc_{}'.format(path,key,gcarc),'{}/all_events/event_{}/waveforms/BHE/gcarc_{}'.format(path,value,gcarc),'{}/pair_events/master_{}/egf_{}/S/gcarc_{}'.format(path,key,value,gcarc)])
        subprocess.call(['bash','/home/meichen/bin/add_BHNE.sh','{}/pair_events/master_{}/egf_{}/S/gcarc_{}'.format(path,key,value,gcarc)])
        subprocess.call(['python','/home/meichen/bin/seismoscript_mtspec.py',"*.master","t2",'{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'master'])
        subprocess.call(['python','/home/meichen/bin/seismoscript_mtspec.py',"*.egf","t2",'{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'egf'])
    subprocess.call(['bash','/home/meichen/bin/sr_calc.sh','{}'.format(window_begin),'{}'.format(window_length),'{}'.format(window_multi),'{}'.format(phase),'{}'.format(gcarc)])
        
    os.chdir('{}/'.format(path))
    

def mk_path(path):
    import os
    import subprocess
    isExist=os.path.exists(path)
    if isExist:
        subprocess.call(['rm -r {}'.format(path)],shell=True)
    os.makedirs(path)

def download_data(event_info,save_path):
    import obspy
    from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader

    lat=event_info[2]
    lon=event_info[3]
    origin_time=obspy.UTCDateTime(event_info[1])

    domain=CircularDomain(latitude=lat,longitude=lon,minradius=0.0,maxradius=85.0)
    restrictions=Restrictions(starttime=origin_time - 5*60,endtime=origin_time + 30*60,reject_channels_with_gaps=True,minimum_length=0.95,minimum_interstation_distance_in_m=10E2,channel_priorities=["BH[ZNE]"],location_priorities=["","00","10"])

    mdl=MassDownloader(providers=["IRIS"])
    mdl.download(domain,restrictions,mseed_storage="{}/waveforms".format(save_path),stationxml_storage="{}/stations".format(save_path))

def mseed2sac(path):
    import subprocess
    for stnxml in glob.glob('{}/stations/*'.format(path)):
        stationname=stnxml.split('/')[-1]
        nw=stationname.split('.')[0]
        stn=stationname.split('.')[1]
        subprocess.call(['java','-jar','/home/meichen/bin/stationxml-seed-converter-2.0.4-SNAPSHOT.jar','--input','{}/stations/{}.{}.xml'.format(path,nw,stn),'--output','{}/waveforms/{}.{}.dataless'.format(path,nw,stn)])
        for filename in glob.glob('{}/waveforms/{}.{}.*.mseed'.format(path,nw,stn)):
            mseedfile=filename.split('/')[-1]
            subprocess.call(['rdseed','-df','{}/waveforms/{}'.format(path,mseedfile),'-z','1','-g','{}/waveforms/{}.{}.dataless'.format(path,nw,stn),'-q','{}/waveforms/'.format(path)])

main()
#test = fit(main_path="/home/jritsema/work1/test/pair_events",master=5149896,egf=5154168,phase="S",dist_range="85",window_length=40)
#print(test)
