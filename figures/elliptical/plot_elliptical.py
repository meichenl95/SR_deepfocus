#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special

def source_a(depth):
    depth1 = 400 # unit is km
    depth2 = 700
    thick1 = 5 # thickness of wedge at depth of 400 km
    thick2 = 0
    return (depth2 - depth)/(depth2 - depth1)*(thick1 - thick2)*1000 # return in unit meter

def stress_drop(fc,M0,PREM,depth):
    radius = 6371000-depth*1000
    temp = (PREM[0]<radius).sum()
    Vs = (radius-PREM[0][temp-1])/(PREM[0][temp]-PREM[0][temp-1])*(PREM[3][temp]-PREM[3][temp-1])+PREM[3][temp-1]
    dimension_a = source_a(depth)

    ks = 0.28
    dimension_b = (Vs*ks/fc)**2/dimension_a
    k = np.sqrt(1-(min(dimension_a,dimension_b)/max(dimension_a,dimension_b))**2)
    E_k = special.ellipe(k)
    K_k = special.ellipk(k)
    C = 4./(3*E_k + (E_k-(1-k**2)*K_k)/k/k)
    print(depth,dimension_a,dimension_b,C)
    return M0/C/(np.pi*dimension_a*dimension_b*min(dimension_a,dimension_b))

def tao2fc(tao,M0,depth,PREM):
    radius = 6371000-depth*1000
    temp = (PREM[0]<radius).sum()
    Vs = 6000
    dimension_a = source_a(depth)
    ks = 0.28

    dimension_b = M0/(np.pi*dimension_a*dimension_a*tao)
    if dimension_b<dimension_a:
        dimension_b = M0/(np.pi*dimension_a*tao)**0.5
    return ks*Vs/np.sqrt(dimension_a*dimension_b)

def M02mag(M0):
    return 2./3. * np.log10(M0) - 6.03

def mag2M0(mag):
    return 10**(1.5*(mag+6.03))

def M0r2magdif(M0_ratio):
    return 2./3. * np.log10(M0_ratio)

#String to be initialized
phase_distance = 'S_rg'
tao_lb = 0.01*1e6
tao_ub = 100*1e6

fig,ax = plt.subplots(3,4,figsize=[18,12])
## Seperate total EGFs
data = pd.read_csv('{}_total.csv'.format(phase_distance),skipinitialspace=True)
PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)
data_array = np.array(data)
depth = []
fc = []
M0 = []
M0_ratio = []
magdif = []
mag = []
for i in np.arange(len(data_array[:,0])):
    depth.append(data_array[i,5])
    fc.append(data_array[i,17])
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
for i in np.arange(len(fc)):
    tao.append(stress_drop(fc[i],M0[i],PREM,depth[i]))
ax[1,0].plot(depth,tao,linestyle='',marker='o',color='blue',markersize=3)
ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('Depth(km)')
ax[1,0].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,0].plot([1e17,1e23],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,0].plot([1e17,1e23],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,0].plot(mag2M0(np.array(mag)),fc,linestyle='',marker='o',markersize=3,color='blue')
ax_twin = ax[2,0].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot(mag,fc,linestyle='',marker='o',markersize=3,color='blue')
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
ax[0,1].scatter(magdif,fitmagdif,marker='o',color='blue',s=15,alpha=0.3)
ax[0,1].plot([0.5,2.5],[1.0,3.0],linestyle='--',linewidth=2,color='gray')
ax[0,1].plot([0.5,2.5],[0,2.0],linestyle='--',linewidth=2,color='gray')
ax[0,1].set_xlabel('Catalog mag. diff.')
ax[0,1].set_ylabel('Fitting mag. diff.')

# Stress drop VS. Depth
fc_mean_pairs = {}
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
    if i>0 and data_array[i,0] != data_array[i-1,0]:
        depth.append(data_array[i,5])
        M0.append(data_array[i,9])
        mag.append(data_array[i,2])
        lat.append(data_array[i,3])
        lon.append(data_array[i,4])

fc = np.ones(len(fc_mean_pairs))
tao = np.ones(len(fc_mean_pairs))
for counter,key in enumerate(list(fc_mean_pairs.keys())):
    mean = fc_mean_pairs.get(key)
    for i in np.arange(len(mean)):
        fc[counter] = fc[counter]*mean[i]
    fc[counter] = fc[counter]**(1./len(mean))
    tao[counter] = stress_drop(fc[counter],M0[counter],PREM,depth[counter])
ax[1,1].plot(depth,tao,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
ax[1,1].set_yscale('log')
ax[1,1].set_xlabel('Depth(km)')
ax[1,1].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,1].plot([1e17,1e23],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,1].plot([1e17,1e23],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,1].plot(mag2M0(np.array(mag)),fc,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax_twin = ax[2,1].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot(mag,fc,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
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
M0 = []
M0_ratio = []
magdif = []
mag = []
for i in np.arange(len(data_array[:,0])):
    depth.append(data_array[i,5])
    fc.append(data_array[i,17])
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
for i in np.arange(len(fc)):
    tao.append(stress_drop(fc[i],M0[i],PREM,depth[i]))
ax[1,2].plot(depth,tao,linestyle='',marker='o',color='blue',markersize=3)
ax[1,2].set_yscale('log')
ax[1,2].set_xlabel('Depth(km)')
ax[1,2].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,2].plot([1e17,1e23],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,2].plot([1e17,1e23],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,2].plot(mag2M0(np.array(mag)),fc,linestyle='',marker='o',markersize=3,color='blue')
ax_twin = ax[2,2].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot(mag,fc,linestyle='',marker='o',markersize=3,color='blue')
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
    if i>0 and data_array[i,0] != data_array[i-1,0]:
        depth.append(data_array[i,5])
        M0.append(data_array[i,9])
        mag.append(data_array[i,2])
        lat.append(data_array[i,3])
        lon.append(data_array[i,4])

fc = np.ones(len(fc_mean_pairs))
tao = np.ones(len(fc_mean_pairs))
for counter,key in enumerate(list(fc_mean_pairs.keys())):
    mean = fc_mean_pairs.get(key)
    for i in np.arange(len(mean)):
        fc[counter] = fc[counter]*mean[i]
    fc[counter] = fc[counter]**(1./len(mean))
    tao[counter] = stress_drop(fc[counter],M0[counter],PREM,depth[counter])
    print(key,tao[counter],M0[counter],lat[counter],lon[counter])
ax[1,3].plot(depth,tao,linestyle='',marker='o',mfc='red',mec='blue',markersize=3)
ax[1,3].set_yscale('log')
ax[1,3].set_xlabel('Depth(km)')
ax[1,3].set_ylabel('Stress drop(Pa)')

# Corner frequency VS. Seismic moment w/ constant Stress drop lines
ax[2,3].plot([1e17,1e23],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,3].plot([1e17,1e23],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax[2,3].plot(mag2M0(np.array(mag)),fc,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax_twin = ax[2,3].twiny()
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_lb,1e17,600,PREM),tao2fc(tao_lb,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot([M02mag(1e17),M02mag(1e23)],[tao2fc(tao_ub,1e17,600,PREM),tao2fc(tao_ub,1e23,600,PREM)],linestyle='--',color='grey')
ax_twin.plot(mag,fc,linestyle='',marker='o',markersize=3,mfc='red',mec='blue')
ax[2,3].set_xscale('log')
ax[2,3].set_yscale('log')
ax[2,3].set_xlabel('Seismic moment')
ax[2,3].set_ylabel('Corner frequency(Hz)')

fig.tight_layout()
fig.savefig('fig_elliptical.pdf')
