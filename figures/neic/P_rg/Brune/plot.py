#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def stress_drop(fc,M0,PREM,depth):
    # Sato and Hirasawa
    radius = 6371000-depth*1000
    temp = (PREM[0]<radius).sum()
#    Vp = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[2][temp]-PREM[2][temp-1])+PREM[2][temp-1]
#    Cp = 1.6
#    k = Cp / np.pi / 2.

    # Madariaga
    k = 0.32
    Vp = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[3][temp]-PREM[3][temp-1])+PREM[3][temp-1]

    return (7./16.) * (M0*fc**3) / (k**3 * Vp**3)

def M0r2magdif(M0_ratio):
    magdif = []
    for i in np.arange(len(M0_ratio)):
        magdif.append(2./3.*np.log10(M0_ratio[i]))
    return magdif

#String to be initialized
phase_distance = 'rgp'
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)
## Separate certified EGFs
data = pd.read_csv('pairsfile_{}_select.csv'.format(phase_distance),skipinitialspace=True)
data_array = np.array(data)
depth = data_array[:,5]
fc = data_array[:,17]
fc_std = data_array[:,18]*2
M0 = data_array[:,9]
M0_ratio = data_array[:,15]
magdif = data_array[:,10]
mag = data_array[:,2]

## plot figures
fig,ax = plt.subplots(2,4,figsize=[12,6])
# corner frequency VS. magnitude
fc_lower_err = []
fc_upper_err = []
for i in np.arange(len(fc)):
    fc_lower_err.append(fc[i]-10**(np.log10(fc[i])-fc_std[i]))
    fc_upper_err.append(10**(np.log10(fc[i])+fc_std[i])-fc[i])
ax[0,0].errorbar(mag,fc,yerr=np.vstack((fc_lower_err,fc_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[0,0].set_yscale('log')
ax[0,0].set_ylabel('$f_c (Hz)$')
ax[0,0].set_xlabel('Mw')
# corner frequency VS. depth
ax[0,1].errorbar(depth,fc,yerr=np.vstack((fc_lower_err,fc_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[0,1].set_yscale('log')
ax[0,1].set_ylabel('$f_c (Hz)$')
ax[0,1].set_xlabel('Depth (km)')
# Stress drop VS. Depth
tao = []
tao_lower_err = []
tao_upper_err = []
for i in np.arange(len(fc)):
    tao.append(stress_drop(fc[i],M0[i],PREM,depth[i]))
    tao_lower_err.append(tao[i]-stress_drop(fc[i]-fc_lower_err[i],M0[i],PREM,depth[i]))
    tao_upper_err.append(stress_drop(fc[i]+fc_upper_err[i],M0[i],PREM,depth[i])-tao[i])
ax[0,2].errorbar(depth,tao,yerr=np.vstack((tao_lower_err,tao_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[0,2].set_yscale('log')
ax[0,2].set_xlabel('Depth(km)')
ax[0,2].set_ylabel(r'$\Delta\ \tau_P (MPa)$')
# Stress drop VS. magnitude
ax[0,3].errorbar(mag,tao,yerr=np.vstack((tao_lower_err,tao_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[0,3].set_yscale('log')
ax[0,3].set_xlabel('Mw')
ax[0,3].set_ylabel(r'$\Delta\ \tau_P (MPa)$')
# print magnitude diff. VS. catalog magnitude diff.
#fitmagdif = M0r2magdif(M0_ratio)
#for i in np.arange(len(fitmagdif)):
#    print("%.2f %.2f" % (magdif[i],fitmagdif[i]))

## Combined certified EGFs
# Stress drop VS. Depth
fc_pairs = {}
fc_std_pairs = {}
masterid = []
M0 = []
depth = []
mag = []
lat = []
lon = []

for i in np.arange(len(data_array[:,0])):
    if data_array[i,0] not in fc_pairs.keys():
        depth.append(data_array[i,5])
        M0.append(data_array[i,9])
        mag.append(data_array[i,2])
        masterid.append(data_array[i,0])
        lat.append(data_array[i,3])
        lon.append(data_array[i,4])
    fc_pairs.setdefault(data_array[i,0],[]).append(data_array[i,17])
    fc_std_pairs.setdefault(data_array[i,0],[]).append(data_array[i,18]*2)

fc = np.ones(len(masterid))
fc_std = np.zeros(len(masterid))
tao = np.ones(len(masterid))
for j in np.arange(len(masterid)):
    key = masterid[j]
    mean = fc_pairs.get(key)
    std = fc_std_pairs.get(key)
    for i in np.arange(len(mean)):
        fc[j] = fc[j]*mean[i]
        fc_std[j] = fc_std[j] + std[i]**2
    fc[j] = fc[j]**(1./len(mean))
    fc_std[j] = fc_std[j]**0.5
    tao[j] = stress_drop(fc[j],M0[j],PREM,depth[j])

# corner frequency VS. magnitude
fc_lower_err = []
fc_upper_err = []
for i in np.arange(len(fc)):
    fc_lower_err.append(fc[i]-10**(np.log10(fc[i])-fc_std[i]))
    fc_upper_err.append(10**(np.log10(fc[i])+fc_std[i])-fc[i])
ax[1,0].errorbar(mag,fc,yerr=np.vstack((fc_lower_err,fc_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[1,0].set_yscale('log')
ax[1,0].set_ylabel('$f_c (Hz)$')
ax[1,0].set_xlabel('Mw')
# corner frequency VS. depth
ax[1,1].errorbar(depth,fc,yerr=np.vstack((fc_lower_err,fc_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[1,1].set_yscale('log')
ax[1,1].set_ylabel('$f_c (Hz)$')
ax[1,1].set_xlabel('Depth (km)')
# stress drop VS. depth
tao_lower_err = []
tao_upper_err = []
for i in np.arange(len(fc)):
    tao_lower_err.append(tao[i]-stress_drop(fc[i]-fc_lower_err[i],M0[i],PREM,depth[i]))
    tao_upper_err.append(stress_drop(fc[i]+fc_upper_err[i],M0[i],PREM,depth[i])-tao[i])
ax[1,2].errorbar(depth,tao,yerr=np.vstack((tao_lower_err,tao_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[1,2].set_yscale('log')
ax[1,2].set_ylabel(r'$\Delta\ \tau_P (MPa)$')
ax[1,2].set_xlabel('Depth (km)')
# stress drop VS. magnitude
ax[1,3].errorbar(mag,tao,yerr=np.vstack((tao_lower_err,tao_upper_err)),linestyle='',marker='o',color='blue',markersize=3)
ax[1,3].set_yscale('log')
ax[1,3].set_ylabel(r'$\Delta\ \tau_P (MPa)$')
ax[1,3].set_xlabel('Mw')

for j in np.arange(len(masterid)):
    print(masterid[j],tao[j],M0[j],lat[j],lon[j],depth[j],mag[j],fc[j],fc_lower_err[j],fc_upper_err[j],tao_lower_err[j],tao_upper_err[j],(7./16.*M0[j]/tao[j])**(1./3.)/1000.)
#    print(masterid[j],fc[j])
    

with open("fc_logtao_{}.txt".format(phase_distance),'w') as f:
    for i in np.arange(len(lat)):
        f.write("%s\t%s\t%s\t%s\t%s\n" % (fc[i],np.log10(tao[i]),lat[i],lon[i],depth[i]))
f.close()

fig.tight_layout()
fig.savefig('fig.pdf')
