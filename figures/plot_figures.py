#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def tao2fc(tao,M0,V,C):
    k = C / np.pi / 2
    constant = 7./16. / (k**3*V**3)
    return (tao/M0/constant)**(1./3.)
def mag2M0(mag):
    return 10**(1.5*(mag+6.03))
def M02mag(M0):
    return 2./3. * np.log10(M0) - 6.03

data = pd.read_csv('pairsfile_gcmt.csv',skipinitialspace=True)
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)
data_array = np.array(data)
num_stations = 2
std_to_mean = 0.1
upper_fc = 1.98
lower_fc = 0.026

# plot magnitude difference vs. moment mag
fig,ax = plt.subplots(2,3,figsize=[12,7])
for m,phase in zip(np.arange(2),['S','P']):
    for j,distance in zip(np.arange(3),['regional','global','rg']):
        data_array_filter = data_array[data_array[:,14+m*12+j*4]>num_stations]
        data_array_filter = data_array_filter[data_array_filter[:,10]>0]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]<upper_fc]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]>lower_fc]
        data_array_filter = data_array_filter[data_array_filter[:,16+m*12+j*4]/data_array_filter[:,17+m*12+j*4]<std_to_mean]
        magdif = data_array_filter[:,10]
        fit = data_array_filter[:,15+m*12+j*4]
        fit_a = []
        for i in np.arange(len(fit)):
            fit_a.append(float(fit[i].split()[0].split()[0][1::]))
        a_final = []
        mag_final = []
        for count,value in enumerate(fit_a):
            if 1.1 <value < 200:
                a_final.append(value)
                mag_final.append(magdif[count])
        a_final = 2./3.*np.log10(a_final)
        ax[m,j].scatter(mag_final,a_final,s=10,label='{}'.format(len(a_final)))
        ax[m,j].legend()
        ax[m,j].set_xlim([0,2])
        ax[m,j].set_ylim([0,2])
        ax[m,j].set_title('{} {}'.format(distance,phase))
        ax[m,j].set_xlabel('Fitting magdif')
        ax[m,j].set_ylabel('Catalog magdif')

fig.tight_layout()
#plt.suptitle('magnitude difference(x axis) vs.\n fitting moment mag(y axis)')
plt.savefig('mag_difference.png')

# plot distribution of station recording number
fig,ax = plt.subplots(2,3,figsize=[12,7])
for m,phase in zip(np.arange(2),['S','P']):
    for j,distance in zip(np.arange(3),['regional','global','rg']):
        data_array_filter = data_array[data_array[:,14+m*12+j*4]>num_stations]
        data_array_filter = data_array_filter[data_array_filter[:,10]>0]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]<upper_fc]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]>lower_fc]
        data_array_filter = data_array_filter[data_array_filter[:,16+m*12+j*4]/data_array_filter[:,17+m*12+j*4]<std_to_mean]
        stnnm = data_array_filter[:,14+m*12+j*4]
        tmp = []
        for i in np.arange(len(stnnm)):
            tmp.append(stnnm[i])
        median = np.median(tmp)
        ax[m,j].hist(tmp,10,label='median {}'.format(median))
        ax[m,j].legend()
        ax[m,j].set_xlabel('Station')
        ax[m,j].set_ylabel('Number')
        ax[m,j].set_title('{}.{}'.format(distance,phase))

fig.tight_layout()
#plt.suptitle('Distribution of station number of each pair')
plt.savefig('stnnm_hist.png')


# plot stress drop and corner frequency vs. depth
fig,ax = plt.subplots(2,3,figsize=[12,8])
fig2,ax2 = plt.subplots(2,3,figsize=[12,8])
for m,phase in zip(np.arange(2),['S','P']):
    for j,distance in zip(np.arange(3),['regional','global','rg']):
        data_array_filter = data_array[data_array[:,14+m*12+j*4]>num_stations]
        data_array_filter = data_array_filter[data_array_filter[:,10]>0]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]<upper_fc]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]>lower_fc]
        data_array_filter = data_array_filter[data_array_filter[:,16+m*12+j*4]/data_array_filter[:,17+m*12+j*4]<std_to_mean]

        mean_pairs = {}
        std_pairs = {}
        M0_gcmt = []
        depth = []
        depth.append(data_array_filter[0,5])
        M0_gcmt.append(data_array_filter[0,9])
        
        for i in np.arange(len(data_array_filter[:,0])):
            mean_pairs.setdefault(data_array_filter[i,0],[]).append(data_array_filter[i,17+m*12+j*4])
            std_pairs.setdefault(data_array_filter[i,0],[]).append(data_array_filter[i,16+m*12+j*4])
            if i>0 and data_array_filter[i,0] != data_array_filter[i-1,0]:
                depth.append(data_array_filter[i,5])
                M0_gcmt.append(data_array_filter[i,9])

        fc = np.ones(len(mean_pairs))
        fc_std = np.zeros(len(mean_pairs))
        stress_drop = np.ones(len(mean_pairs))
        stress_drop_std = np.zeros(len(mean_pairs))
        
        for counter,key in enumerate(list(mean_pairs.keys())):
            mean = mean_pairs.get(key)
            std = std_pairs.get(key)

            for i in np.arange(len(mean)):
                fc[counter] = fc[counter]*mean[i]
                fc_std[counter] = fc_std[counter] + (std[i]/mean[i])**2

            fc[counter] = fc[counter]**(1/len(mean))
            fc_std[counter] = 1/len(mean)*fc[counter]*fc_std[counter]**0.5
                   
            radius = 6371000-depth[counter]*1000
            temp = (PREM[0]<radius).sum()
            if phase == 'S':
                Vs = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[3][temp]-PREM[3][temp-1])+PREM[3][temp-1]
                stress_drop[counter] = 7*(np.pi**3)*M0_gcmt[counter]*(fc[counter]**3)/(2*(1.99**3)*(Vs**3))*0.000001
                stress_drop_std[counter] = 7*3*(np.pi**3)*M0_gcmt[counter]*(fc[counter]**2)*fc_std[counter]/(2*(1.99**3)*(Vs**3))*0.000001
            elif phase == 'P':
                Vp = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[2][temp]-PREM[2][temp-1])+PREM[2][temp-1]
                stress_drop[counter] = 7*(np.pi**3)*M0_gcmt[counter]*(fc[counter]**3)/(2*(1.6**3)*(Vp**3))*0.000001
                stress_drop_std[counter] = 7*3*(np.pi**3)*M0_gcmt[counter]*(fc[counter]**2)*fc_std[counter]/(2*(1.6**3)*(Vp**3))*0.000001
        ax[m,j].errorbar(depth,stress_drop,yerr=stress_drop_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
        ax[m,j].set_yscale('log')
        ax[m,j].set_xlabel('Depth(km)')
        ax[m,j].set_ylabel('Stress Drop(MPa)')
        ax[m,j].set_title('distance={}, phase={}, points={}'.format(distance,phase,len(depth)))
        ax2[m,j].errorbar(depth,fc,yerr=fc_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
        ax2[m,j].set_yscale('log')
        ax2[m,j].set_xlabel('Depth(km)')
        ax2[m,j].set_ylabel('Corner Frequency(Hz)')
        ax2[m,j].set_title('distance={}. phase={}, points={}'.format(distance,phase,len(depth)))

fig.tight_layout()
fig2.tight_layout()
#fig.suptitle('Event depth(x axis) vs. stress drop(y axis)')
fig.savefig('stressdrop_depth.png')
#fig2.suptitle('Event depth(x axis) vs. corner frequency(y axis)')
fig2.savefig('fc_depth.png')


# plot corner frequency vs. moment magnitude
fig,ax = plt.subplots(2,3,figsize=[12,7])
for m,phase in zip(np.arange(2),['S','P']):
    for j,distance in zip(np.arange(3),['regional','global','rg']):
        data_array_filter = data_array[data_array[:,14+m*12+j*4]>num_stations]
        data_array_filter = data_array_filter[data_array_filter[:,10]>0]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]<upper_fc]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]>lower_fc]
        data_array_filter = data_array_filter[data_array_filter[:,16+m*12+j*4]/data_array_filter[:,17+m*12+j*4]<std_to_mean]

        mean_pairs = {}
        std_pair = {}
        magdif = []
        M0_gcmt = []
        mag = []
        depth = []
        magdif.append(data_array_filter[0,10])
        M0_gcmt.append(data_array_filter[0,9])
        depth.append(data_array_filter[0,5])
        mag.append(2./3.*(np.log10(np.float(M0_gcmt[0])))-9.1)
        
        for i in np.arange(len(data_array_filter[:,0])):
            mean_pairs.setdefault(data_array_filter[i,0],[]).append(data_array_filter[i,17+m*12+j*4])
            std_pairs.setdefault(data_array_filter[i,0],[]).append(data_array_filter[i,16+m*12+j*4])
            if i>0 and data_array_filter[i,0] !=data_array_filter[i-1,0]:
                M0_gcmt.append(data_array_filter[i,9])
                magdif.append(data_array_filter[i,10])
                depth.append(data_array_filter[i,5])
                mag.append(2./3.*(np.log10(data_array_filter[i,9])-9.1))

        fc = np.ones(len(mean_pairs))
        fc_std = np.zeros(len(mean_pairs))

        for counter,key in enumerate(list(mean_pairs.keys())):
            mean = mean_pairs.get(key)
            std = std_pairs.get(key)

            for i in np.arange(len(mean)):
                fc[counter] = fc[counter]*mean[i]
                fc_std[counter] = fc_std[counter] + (std[i]/mean[i])**2
            fc[counter] = fc[counter]**(1/len(mean))
            fc_std[counter] = 1/len(mean)*fc[counter]*fc_std[counter]**0.5

        ax2 = ax[m,j].twiny()
        sd_low = 0.01*1e6
        sd_up = 300*1e6

        radius = 6371000-depth[counter]*1000
        temp = (PREM[0]<radius).sum()
        if phase == 'S':
            wave_v = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[3][temp]-PREM[3][temp-1])+PREM[3][temp-1]
            C = 1.99
        elif phase == 'P':
            wave_v = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[2][temp]-PREM[2][temp-1])+PREM[2][temp-1]
            C = 1.6
        ax[m,j].plot([1e16,1e23],[tao2fc(sd_low,1e16,wave_v,C),tao2fc(sd_low,1e23,wave_v,C)],linestyle='--',color='orange',label='{}Mpa'.format(sd_low*1e-6))
        ax[m,j].plot([1e16,1e23],[tao2fc(sd_up,1e16,wave_v,C),tao2fc(sd_up,1e23,wave_v,C)],linestyle='--',color='orange',label='{}Mpa'.format(sd_up*1e-6))
        ax[m,j].errorbar(mag2M0(np.array(mag)),fc,yerr=fc_std,linestyle='',marker='o',mfc='red',mec='green',markersize=3)        
        ax2.errorbar(mag,fc,yerr=fc_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)        
        ax2.plot([M02mag(1e16),M02mag(1e23)],[tao2fc(sd_low,1e16,wave_v,C),tao2fc(sd_low,1e23,wave_v,C)],linestyle='--',color='red',label='{}Mpa'.format(sd_low*1e-6))
        ax2.plot([M02mag(1e16),M02mag(1e23)],[tao2fc(sd_up,1e16,wave_v,C),tao2fc(sd_up,1e23,wave_v,C)],linestyle='--',color='red',label='{}Mpa'.format(sd_up*1e-6))
        ax[m,j].set_yscale("log")
        ax[m,j].set_xscale("log")
        ax[m,j].set_xlabel('Seismic moment')
        ax[m,j].set_ylabel('Corner frequency(Hz)')
        ax[m,j].legend()
        ax[m,j].set_title('distance={}, phase={}, points={}'.format(distance,phase,len(fc)),y=1.1)

fig.tight_layout()
#plt.suptitle('moment magnitude(x axis) vs.\ncorner frequency(Hz)')
plt.savefig('fc.png')
        

# plot corner frequency of S vs. P
fig,ax = plt.subplots(1,3,figsize=[12,4])
for j,distance in zip(np.arange(3),['regional','global','rg']):
    data_array_filter_s = data_array[data_array[:,14+j*4]>num_stations]
    data_array_filter_p = data_array[data_array[:,26+j*4]>num_stations]
    data_array_filter_s = data_array_filter_s[data_array_filter_s[:,10]>0]
    data_array_filter_p = data_array_filter_p[data_array_filter_p[:,10]>0]
    data_array_filter_s = data_array_filter_s[data_array_filter_s[:,17+j*4]<upper_fc]
    data_array_filter_s = data_array_filter_s[data_array_filter_s[:,17+j*4]>lower_fc]
    data_array_filter_p = data_array_filter_p[data_array_filter_p[:,29+j*4]<upper_fc]
    data_array_filter_p = data_array_filter_p[data_array_filter_p[:,29+j*4]>lower_fc]
    data_array_filter_s = data_array_filter_s[data_array_filter_s[:,16+j*4]/data_array_filter_s[:,17+j*4]<std_to_mean]
    data_array_filter_p = data_array_filter_p[data_array_filter_p[:,28+j*4]/data_array_filter_p[:,29+j*4]<std_to_mean]
    
    s_mean_pairs = {}
    s_std_pairs = {}
    p_mean_pairs = {}
    p_std_pairs = {}

    for i in np.arange(len(data_array_filter_p[:,0])):
        if data_array_filter_p[i,0] in list(data_array_filter_s[:,0]):
            p_mean_pairs.setdefault(data_array_filter_p[i,0],[]).append(data_array_filter_p[i,29+j*4])
            p_std_pairs.setdefault(data_array_filter_p[i,0],[]).append(data_array_filter_p[i,28+j*4])

    for i in np.arange(len(data_array_filter_s[:,0])):
        if data_array_filter_s[i,0] in list(data_array_filter_p[:,0]):
            s_mean_pairs.setdefault(data_array_filter_s[i,0],[]).append(data_array_filter_s[i,17+j*4])
            s_std_pairs.setdefault(data_array_filter_s[i,0],[]).append(data_array_filter_s[i,16+j*4])

    s_fc = np.ones(len(s_mean_pairs))
    s_fc_std = np.zeros(len(s_mean_pairs))
    p_fc = np.ones(len(p_mean_pairs))
    p_fc_std = np.zeros(len(p_mean_pairs))
#    for key in s_mean_pairs.keys():
#        index = list(data_array_filter_s[:,0]).index(key)
#        print("S",key,data_array_filter_s[index,3],data_array_filter_s[index,4],s_mean_pairs.get(key))
#    for key in p_mean_pairs.keys():
#        index = list(data_array_filter_p[:,0]).index(key)
#        print("P",key,data_array_filter_p[index,3],data_array_filter_p[index,4],p_mean_pairs.get(key))

    for counter,key in enumerate(list(s_mean_pairs.keys())):
        s_mean = s_mean_pairs.get(key)
        s_std = s_std_pairs.get(key)
        p_mean = p_mean_pairs.get(key)
        p_std = p_std_pairs.get(key)

        for i in np.arange(len(s_mean)):
            s_fc[counter] = s_fc[counter]*s_mean[i]
            s_fc_std[counter] = s_fc_std[counter] + (s_std[i]/s_mean[i])**2
        for i in np.arange(len(p_mean)):
            p_fc[counter] = p_fc[counter]*p_mean[i]
            p_fc_std[counter] = p_fc_std[counter] + (p_std[i]/p_mean[i])**2
        s_fc[counter] = s_fc[counter]**(1/len(s_mean))
        s_fc_std[counter] = 1/len(s_mean)*s_fc[counter]*s_fc_std[counter]**0.5
        p_fc[counter] = p_fc[counter]**(1/len(p_mean))
        p_fc_std[counter] = 1/len(p_mean)*p_fc[counter]*p_fc_std[counter]**0.5
    ax[j].errorbar(s_fc,p_fc,yerr=p_fc_std,xerr=s_fc_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3,label='{}'.format(len(s_fc)))
    ax[j].plot([0,4],[0,4],color='orange')
    ax[j].legend()
    ax[j].set_xlim([0,4])
    ax[j].set_ylim([0,4])
    ax[j].set_xlabel('Corner frequency of S waves(Hz)')
    ax[j].set_ylabel('Corner frequency of P waves(Hz)')
    ax[j].set_title('{}'.format(distance))
    
fig.tight_layout()
#plt.suptitle('corner frequency of S(x axis) vs. P(y axis)')
plt.savefig('fcs_fcp.png')


# plot std_to_mean vs. moment magnitude
fig,ax = plt.subplots(2,3,figsize=[12,7])
fig2,ax2 = plt.subplots(2,3,figsize=[12,7])
for m,phase in zip(np.arange(2),['S','P']):
    for j,distance in zip(np.arange(3),['regional','global','rg']):
        data_array_filter = data_array[data_array[:,14+m*12+j*4]>num_stations]
        data_array_filter = data_array_filter[data_array_filter[:,10]>0]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]<upper_fc]
        data_array_filter = data_array_filter[data_array_filter[:,17+m*12+j*4]>lower_fc]
        std_to_mean = data_array_filter[:,16+m*12+j*4]/data_array_filter[:,17+m*12+j*4]
        mastermag = []
        egfmag = []
        for i in np.arange(len(std_to_mean)):
            mastermag.append(float(str(data_array_filter[:,2][i])[-3::]))
            egfmag.append(float(str(data_array_filter[:,8][i])[-3::]))
        ax[m,j].scatter(mastermag,std_to_mean,s=10,label='{}.'.format(len(mastermag)))
        ax[m,j].set_xlabel('Moment magnitude(master)')
        ax[m,j].set_ylabel('std_to_mean')
        ax[m,j].legend()
        ax2[m,j].scatter(egfmag,std_to_mean,s=10,label='{}.'.format(len(egfmag)))
        ax2[m,j].set_xlabel('Moment magnitude(egf)')
        ax2[m,j].set_ylabel('std_to_mean')
        ax2[m,j].legend()

fig.tight_layout()
fig2.tight_layout()
#fig.suptitle('master event moment magnitude(x axis)\nvs. std_to_mean of bootstrap(y axis)')
#fig2.suptitle('egf event moment magnitude(x axis)\nvs. std_to_mean of bootstrap(y axis)')
fig.savefig('mastermag.png')
fig2.savefig('egfmag.png')

