#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('pairsfile_rgp_select.csv',skipinitialspace=True)
data_array = np.array(data)
master_lat = []
master_lon = []
master_radius = []
master_mag = []
egf_lat = []
egf_lon = []
egf_radius = []
egf_mag = []
dist = []
for i in np.arange(len(data_array[:,0])):
    master_lat.append(np.float(data_array[i,3])*np.pi/180.0)
    master_lon.append(np.float(data_array[i,4])*np.pi/180.0)
    master_radius.append(6371-np.float(data_array[i,5]))
    master_mag.append(np.float(data_array[i,2]))
    egf_mag.append(np.float(data_array[i,8]))
    egf_lat.append(np.float(data_array[i,11])*np.pi/180.0)
    egf_lon.append(np.float(data_array[i,12])*np.pi/180.0)
    egf_radius.append(6371-np.float(data_array[i,13]))

for i in np.arange(len(master_lat)):
    arg = np.cos(master_lat[i])*np.cos(egf_lat[i])*np.cos(master_lon[i]-egf_lon[i])+np.sin(master_lat[i])*np.sin(egf_lat[i])
    d = np.sqrt(master_radius[i]**2+egf_radius[i]**2-2*master_radius[i]*egf_radius[i]*arg)
    dist.append(d)

print(np.sum(np.array(dist)<300)*1.0/len(dist))

fig,ax=plt.subplots(1,2,figsize=[10,5])
ax[0].hist(dist,bins=15,edgecolor='k',facecolor='gray',lw=0.1,alpha=0.7)
ax[0].tick_params(labelsize=8)
ax[0].set_xlabel('Hypocentral Distance (km)',size=10)
ax[0].set_ylabel('Number',size=10)
ax[1].hist(np.array(master_mag)-np.array(egf_mag),bins=15,edgecolor='k',facecolor='gray',lw=0.1,alpha=0.7)
ax[1].tick_params(labelsize=8)
ax[1].set_xlabel('Magnitude Difference',size=10)
ax[1].set_ylabel('Number',size=10)
fig.savefig('dist_P.pdf')
