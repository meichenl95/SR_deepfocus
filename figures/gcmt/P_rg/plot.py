#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def stress_drop(fc,M0):
    Vs = 9200. # unit in m/s
    Cs = 1.6
    k = Cs / np.pi / 2.
    return (7./16.) * (M0*fc**3) / (k**3 * Vs**3)

def tao2fc(tao,M0):
    Vs = 9200.
    Cs = 1.6
    k = Cs / np.pi / 2.
    constant = 7./16. / (k**3*Vs**3)
    return (tao/M0/constant)**(1./3.)

def stress_drop_std(fc,M0,fc_std):
    Vs = 9200.
    Cs = 1.6
    k = Cs / np.pi / 2.
    return (7./16.) * (3.*M0*fc**2) / (k**3 * Vs**3) * fc_std

def M02mag(M0):
    return 2./3. * np.log10(M0) - 6.03

def mag2M0(mag):
    return 10**(1.5*(mag+6.03))

def M0r2magdif(M0_ratio):
    return 2./3. * np.log10(M0_ratio)

#String to be initialized
phase_distance = 'P_rg'
tao_lb = 0.01*1e6
tao_ub = 100*1e6

fig,ax = plt.subplots(3,4,figsize=[18,12])
## Seperate total EGFs
data = pd.read_csv('{}_total.csv'.format(phase_distance),skipinitialspace=True)
data_array = np.array(data)
depth = []
fc = []
fc_std = []
M0 = []
M0_ratio = []
magdif = []
mag = []
for i in np.arange(len(data_array[:,0])):
    depth.append(data_array[i,5])
    fc.append(data_array[i,17])
    fc_std.append(data_array[i,16])
    M0.append(data_array[i,9])
    M0_ratio.append(float(data_array[i,15].split()[0].split()[0][1::]))
    magdif.append(data_array[i,10])
    mag.append(data_array[i,2])
    if np.abs(data_array[i,10]-M0r2magdif(float(data_array[i,15].split()[0].split()[0][1::]))) > 0.5:
        print(data_array[i,0],data_array[i,6])

# Fitting magnitude diff. VS. Catalog magnitude diff.
fitmagdif = M0r2magdif(M0_ratio)
ax[0,0].scatter(magdif,fitmagdif,marker='o',color='blue',s=10)
ax[0,0].set_xlabel('Catalog mag. diff.')
ax[0,0].set_ylabel('Fitting mag. diff.')

# Stress drop VS. Depth
tao = []
tao_std = []
for i in np.arange(len(fc)):
    tao.append(stress_drop(fc[i],M0[i]))
    tao_std.append(stress_drop_std(fc[i],M0[i],fc_std[i]))
ax[1,0].errorbar(depth,tao,yerr=tao_std,linestyle='',marker='o',color='blue',markersize=3)
ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('Depth(km)')
ax[1,0].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,0].plot([1e17,1e23],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax[2,0].plot([1e17,1e23],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax[2,0].errorbar(mag2M0(np.array(mag)),fc,yerr=fc_std,linestyle='',marker='o',markersize=3,color='blue')
ax_twin = ax[2,0].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax_twin.errorbar(mag,fc,yerr=fc_std,linestyle='',marker='o',markersize=3,color='blue')
ax[2,0].set_xscale('log')
ax[2,0].set_yscale('log')
ax[2,0].set_xlabel('Seismic moment')
ax[2,0].set_ylabel('Corner frequency(Hz)')


## Combined total EGFs
data = pd.read_csv('{}_total.csv'.format(phase_distance),skipinitialspace=True)
data_array = np.array(data)

# Fitting magnitude diff. VS. Catalog magnitude diff.
M0_ratio = []
magdif = []
for i in np.arange(len(data_array[:,0])):
    M0_ratio.append(float(data_array[i,15].split()[0].split()[0][1::]))
    magdif.append(data_array[i,10])
fitmagdif = M0r2magdif(M0_ratio)
ax[0,1].scatter(magdif,fitmagdif,marker='o',color='blue',s=10)
ax[0,1].set_xlabel('Catalog mag. diff.')
ax[0,1].set_ylabel('Fitting mag. diff.')

# Stress drop VS. Depth
fc_mean_pairs = {}
fc_std_pairs = {}
M0 = []
depth = []
mag = []
lat = []
lon = []
depth.append(data_array[0,5])
M0.append(data_array[0,9])
mag.append(data_array[0,2])
lat.append(data_array[0,3])
lon.append(data_array[0,4])

for i in np.arange(len(data_array[:,0])):
    fc_mean_pairs.setdefault(data_array[i,0],[]).append(data_array[i,17])
    fc_std_pairs.setdefault(data_array[i,0],[]).append(data_array[i,16])
    if i>0 and data_array[i,0] != data_array[i-1,0]:
        depth.append(data_array[i,5])
        M0.append(data_array[i,9])
        mag.append(data_array[i,2])
        lat.append(data_array[i,3])
        lon.append(data_array[i,4])

fc = np.ones(len(fc_mean_pairs))
fc_std = np.zeros(len(fc_mean_pairs))
tao = np.ones(len(fc_mean_pairs))
tao_std = np.zeros(len(fc_mean_pairs))
for counter,key in enumerate(list(fc_mean_pairs.keys())):
    mean = fc_mean_pairs.get(key)
    std = fc_std_pairs.get(key)
    for i in np.arange(len(mean)):
        fc[counter] = fc[counter]*mean[i]
        fc_std[counter] = fc_std[counter] + (std[i]/mean[i])**2
    fc[counter] = fc[counter]**(1./len(mean))
    fc_std[counter] = 1./len(mean)*fc[counter] * fc_std[counter]**0.5
    tao[counter] = stress_drop(fc[counter],M0[counter])
    tao_std[counter] = stress_drop_std(fc[counter],M0[counter],fc_std[counter])
ax[1,1].errorbar(depth,tao,yerr=tao_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
ax[1,1].set_yscale('log')
ax[1,1].set_xlabel('Depth(km)')
ax[1,1].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,1].plot([1e17,1e23],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax[2,1].plot([1e17,1e23],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax[2,1].errorbar(mag2M0(np.array(mag)),fc,yerr=fc_std,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax_twin = ax[2,1].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax_twin.errorbar(mag,fc,yerr=fc_std,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax[2,1].set_xscale('log')
ax[2,1].set_yscale('log')
ax[2,1].set_xlabel('Seismic moment')
ax[2,1].set_ylabel('Corner frequency(Hz)')

with open("fc_logtao_total_{}.txt".format(phase_distance),'w') as f:
    for i in np.arange(len(lat)):
        f.write("%s\t%s\t%s\t%s\n" % (fc[i],np.log10(tao[i]),lat[i],lon[i]))
f.close()


## Separate certified EGFs
data = pd.read_csv('{}_certified.csv'.format(phase_distance),skipinitialspace=True)
data_array = np.array(data)
depth = []
fc = []
fc_std = []
M0 = []
M0_ratio = []
magdif = []
mag = []
for i in np.arange(len(data_array[:,0])):
    depth.append(data_array[i,5])
    fc.append(data_array[i,17])
    fc_std.append(data_array[i,16])
    M0.append(data_array[i,9])
    M0_ratio.append(float(data_array[i,15].split()[0].split()[0][1::]))
    magdif.append(data_array[i,10])
    mag.append(data_array[i,2])

# Fitting magnitude diff. VS. Catalog magnitude diff.
fitmagdif = M0r2magdif(M0_ratio)
ax[0,2].scatter(magdif,fitmagdif,marker='o',color='blue',s=10)
ax[0,2].set_xlabel('Catalog mag. diff.')
ax[0,2].set_ylabel('Fitting mag. diff.')

# Stress drop VS. Depth
tao = []
tao_std = []
for i in np.arange(len(fc)):
    tao.append(stress_drop(fc[i],M0[i]))
    tao_std.append(stress_drop_std(fc[i],M0[i],fc_std[i]))
ax[1,2].errorbar(depth,tao,yerr=tao_std,linestyle='',marker='o',color='blue',markersize=3)
ax[1,2].set_yscale('log')
ax[1,2].set_xlabel('Depth(km)')
ax[1,2].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,2].plot([1e17,1e23],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax[2,2].plot([1e17,1e23],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax[2,2].errorbar(mag2M0(np.array(mag)),fc,yerr=fc_std,linestyle='',marker='o',markersize=3,color='blue')
ax_twin = ax[2,2].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax_twin.errorbar(mag,fc,yerr=fc_std,linestyle='',marker='o',markersize=3,color='blue')
ax[2,2].set_xscale('log')
ax[2,2].set_yscale('log')
ax[2,2].set_xlabel('Seismic moment')
ax[2,2].set_ylabel('Corner frequency(Hz)')


## Combined certified EGFs
data = pd.read_csv('{}_certified.csv'.format(phase_distance),skipinitialspace=True)
data_array = np.array(data)

# Fitting magnitude diff. VS. Catalog magnitude diff.
M0_ratio = []
magdif = []
for i in np.arange(len(data_array[:,0])):
    M0_ratio.append(float(data_array[i,15].split()[0].split()[0][1::]))
    magdif.append(data_array[i,10])
fitmagdif = M0r2magdif(M0_ratio)
ax[0,3].scatter(magdif,fitmagdif,marker='o',color='blue',s=10)
ax[0,3].set_xlabel('Catalog mag. diff.')
ax[0,3].set_ylabel('Fitting mag. diff.')

# Stress drop VS. Depth
fc_mean_pairs = {}
fc_std_pairs = {}
M0 = []
depth = []
mag = []
lat = []
lon = []
depth.append(data_array[0,5])
M0.append(data_array[0,9])
mag.append(data_array[0,2])
lat.append(data_array[0,3])
lon.append(data_array[0,4])

for i in np.arange(len(data_array[:,0])):
    fc_mean_pairs.setdefault(data_array[i,0],[]).append(data_array[i,17])
    fc_std_pairs.setdefault(data_array[i,0],[]).append(data_array[i,16])
    if i>0 and data_array[i,0] != data_array[i-1,0]:
        depth.append(data_array[i,5])
        M0.append(data_array[i,9])
        mag.append(data_array[i,2])
        lat.append(data_array[i,3])
        lon.append(data_array[i,4])

fc = np.ones(len(fc_mean_pairs))
fc_std = np.zeros(len(fc_mean_pairs))
tao = np.ones(len(fc_mean_pairs))
tao_std = np.zeros(len(fc_mean_pairs))
for counter,key in enumerate(list(fc_mean_pairs.keys())):
    mean = fc_mean_pairs.get(key)
    std = fc_std_pairs.get(key)
    for i in np.arange(len(mean)):
        fc[counter] = fc[counter]*mean[i]
        fc_std[counter] = fc_std[counter] + (std[i]/mean[i])**2
    fc[counter] = fc[counter]**(1./len(mean))
    fc_std[counter] = 1./len(mean)*fc[counter] * fc_std[counter]**0.5
    tao[counter] = stress_drop(fc[counter],M0[counter])
    tao_std[counter] = stress_drop_std(fc[counter],M0[counter],fc_std[counter])
    print(key,tao[counter],M0[counter])
ax[1,3].errorbar(depth,tao,yerr=tao_std,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
ax[1,3].set_yscale('log')
ax[1,3].set_xlabel('Depth(km)')
ax[1,3].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,3].plot([1e17,1e23],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax[2,3].plot([1e17,1e23],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax[2,3].errorbar(mag2M0(np.array(mag)),fc,yerr=fc_std,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax_twin = ax[2,3].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17),tao2fc(tao_lb,1e23)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17),tao2fc(tao_ub,1e23)],linestyle='--',color='grey')
ax_twin.errorbar(mag,fc,yerr=fc_std,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax[2,3].set_xscale('log')
ax[2,3].set_yscale('log')
ax[2,3].set_xlabel('Seismic moment')
ax[2,3].set_ylabel('Corner frequency(Hz)')

with open("fc_logtao_certified_{}.txt".format(phase_distance),'w') as f:
    for i in np.arange(len(lat)):
        f.write("%s\t%s\t%s\t%s\n" % (fc[i],np.log10(tao[i]),lat[i],lon[i]))
f.close()

fig.tight_layout()
fig.savefig('fig.png')
