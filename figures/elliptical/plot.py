#!/home/meichen/anaconda3/bin/python

def linear_func(x,a,b):
    return a*x +b

def fc2radius(**kwargs):

##-------------------------------------##

# This function calculates the source radius in unit of m from corner frequency
# assuming circular fault model. 

# Created by Meichen Liu on June 21st, 2019

##-------------------------------------##

##parameters
# fc		corner frequency
# depth		the depth of earthquake
# wave		decide the value of k
# model		velocity model

    import numpy as np

    fc = kwargs.get('fc')
    depth = kwargs.get('depth')
    wave = kwargs.get('wave')
    model = kwargs.get('model')
    V = depth2v(depth=depth,wave=wave,model=model)
    if wave == 'S':
        C = 1.99
    elif wave == 'P':
        C = 1.6

    k = C/2./np.pi

    r = k*V/(fc**3)
    return r


def depth2v(**kwargs):

##-------------------------------------##

# This function transer the depth to phase velocity. Returned velocity is in
# m/s.

# Created by Meichen Liu on June. 21st, 2019

##-------------------------------------##

##parameters
# depth		The depth of event in km
# wave		the desired wave
# model		the velocity model

    import pandas as pd

    depth = kwargs.get('depth')
    wave = kwargs.get('wave')
    model = kwargs.get('model')

    radius = 6371000 - depth*1000
    temp = (model[0]<radius).sum()
    if wave == 'S':
        n = 3
    if wave == 'P':
        n = 2
    V = (radius - model[0][temp-1])/(model[0][temp]-model[0][temp-1])*(model[n][temp]-model[n][temp-1])+model[n][temp-1]

    return V

def ellip_tao(**kwargs):

##-------------------------------------##

# This function calculates stress drop assuming elliptical cracks, on the base
# of Allmann and Shearer's results.

# Created by Meichen Liu on June 21st, 2019

##-------------------------------------##

##parameters
# M0		seismic moment
# W		the minor axis of the elliptical crack in km
# L		the major axis of the elliptical crack in km

    import numpy as np
    from scipy import special

    M0 = kwargs.get('M0')
    W  = kwargs.get('W')
    L = kwargs.get('L')

    m = np.sqrt(1-W*W/L/L)
    E_m = special.ellipe(m)
    K_m = special.ellipk(m)
    C = 4./(3.*E_m + (E_m-W*W/L/L*K_m)/m/m)
    tao = M0/(C*np.pi*W*L*W)

    return tao


def cir_tao(**kwargs):

##-------------------------------------##

# This function calculate the stress drop of circular cracks

# Created by Meichen Liu on June 23rd, 2019

##-------------------------------------##

##parameters
# fc		corner frequency
# wave		to determine k
# M0		seismic moment
# depth		event depth
# model		velocity model

    import numpy as np

    fc = kwargs.get('fc')
    wave = kwargs.get('wave')
    M0 = kwargs.get('M0')
    depth = kwargs.get('depth')
    model = kwargs.get('model')

    if wave == 'S':
        C = 1.99
    elif wave == 'P':
        C = 1.6

    k = C/2./np.pi
    V = depth2v(wave=wave,model=model,depth=depth)
    r = k*V/fc
    tao = 7./16.*M0/(r**3)

    return tao    


def width_length(**kwargs):

##-------------------------------------##

# This function decide the other axis length giving one axis length, and return
# the minor and major length in order

# Created by Meichen Liu on June 22nd, 2019

##-------------------------------------##

##parameters
# ax1		Given one axis length
# wave		the wave applied
# fc		corner frequency
# depth		the event depth

    ax1 = kwargs.get('ax1')
    wave = kwargs.get('wave')
    fc = kwargs.get('fc')
    depth = kwargs.get('depth')
    model = kwargs.get('model')

    beta = depth2v(depth=depth,wave='S',model=model)
    if wave == 'S':
        k = 0.28
    elif wave == 'P':
        k = 0.42

    ax2 = (k*beta/fc)**2/ax1
    if ax1 > ax2:
        W = ax2
        L = ax1
    else:
        W = ax1
        L = ax2

    return W,L
    
def slab_thk(**kwargs):

##-------------------------------------##

# This function decide the thickness of slab according to slab2 model from
# USGS. Return the thickness of the subduction slab in unit of m if the 
# information is included in Slab2, else return 0.

# Created by Meichen Liu on June 22nd, 2019

##-------------------------------------##

##parameters
# lat		the latitude of the event
# lon		the longitude of the event

    import pandas as pd
    import numpy as np

    lat = kwargs.get('lat')
    lon = kwargs.get('lon')

    slab2 = pd.read_csv('slab2_thk.xyz',skipinitialspace=True,header=None)
    slab2_array = np.array(slab2)

    lat = round(round(lat/0.05)*0.05,2)
    lon = round(round(lon/0.05)*0.05,2)

    thk = 0
    for i in np.arange(len(slab2_array[:,0])):
        if slab2_array[i,0] == lon:
            if slab2_array[i,1] == lat:
                if slab2_array[i,2] > 0:
                    thk = slab2_array[i,2]

    return thk*1000
    
def geometrical_means(**kwargs):

##-------------------------------------##

# This function calculate geometrical means of those with multiple values. For
# example, master events with multiple eGfs.

# Created by Meichen Liu on June 22nd, 2019

##-------------------------------------##

##parameters
# masterid		a list of the id of the master variable
# v			a list of values to combine

    import numpy as np

    masterid = kwargs.get('masterid')
    v = kwargs.get('v')

    mean_pairs = {}
    for i in np.arange(len(masterid)):
        mean_pairs.setdefault(masterid[i],[]).append(v[i])

    master_return = []
    v_return = np.ones(len(mean_pairs))
    for counter,key in enumerate(list(mean_pairs.keys())):
        master_return.append(key)
        mean = mean_pairs.get(key)
        for i in np.arange(len(mean)):
            v_return[counter] = v_return[counter]*mean[i]
        v_return[counter] = v_return[counter]**(1./len(mean))

    return np.vstack((np.array(master_return),np.array(v_return))).T

def combine_list(**kwargs):

##-------------------------------------##

# This function combine two lists with the same master id (first column), but
# not in the same order. A combined list will be returned.

# Created by Meichen Liu on June 22nd, 2019

##-------------------------------------##

##parameters
# list1			The order of returned list will be the same as list1
# list2			The list to be combined

    import numpy as np

    list1 = kwargs.get('list1')
    list2 = kwargs.get('list2')

    ncol1 = len(list1[0])
    ncol2 = len(list2[0])
    nrow1 = len(list1)
    nrow2 = len(list2)
    if nrow1 != nrow2:
        print("Lengths of two lists differ.")
        exit()

    list_return = np.array([[0.0]*(ncol1+ncol2-1)]*nrow1)
    for i in np.arange(nrow1):
        for j in np.arange(ncol1):
            list_return[i,j] = list1[i,j]
        index = list(list2[:,0]).index(list1[i,0])
        for j in np.arange(ncol2-1):
            list_return[i,j+ncol1] = list2[index,j+1]
    return list_return
        
def drop_duplicate(**kwargs):

##-------------------------------------##

# This function drop duplicate rows

# Created by Meichen Liu on June 22nd, 2019

##-------------------------------------##

##parameters
# mt		a list or an array
# n		the column to check

    import numpy as np

    mt = kwargs.get('mt')
    n = kwargs.get('n')

    nrow = len(mt)
    ncol = len(mt[0])

    mt_uniq = []
    mt_uniq.append(mt[0])
    for i in np.arange(nrow):
        if mt[i][n-1] not in np.array(mt_uniq)[:,n-1]:
            mt_uniq.append(mt[i][:])
    return np.array(mt_uniq)

def main():

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    wave = 'P'
    data = pd.read_csv('pairsfile_rgp_select.csv',skipinitialspace=True)
    data_array = np.array(data)
    PREM = pd.read_csv('PREM_ANISOTROPIC.csv',skipinitialspace=True,header=None)
    if wave == 'S':
        marker = 'd'
        color = 'orange'
    elif wave == 'P':
        marker = 'o'
        color = 'magenta'
    
    eventid = data_array[:,0]
    depth_M0_lat_lon_mag = np.column_stack((data_array[:,5],data_array[:,9],data_array[:,3],data_array[:,4],data_array[:,2]))
    fc = data_array[:,17]

    # calculate geometrical mean corner frequencies of those with multiple eGfs
    geomean = geometrical_means(masterid=eventid,v=fc)
    event_info = drop_duplicate(mt=np.column_stack((eventid,depth_M0_lat_lon_mag)),n=1)
    combined_results = combine_list(list1=geomean,list2=event_info)

    # First method: define length and width assuming circular cracks
    tao1 = []
    radius = []
    for i in np.arange(len(combined_results[:,0])):
        r = fc2radius(fc=float(combined_results[i,1]),wave=wave,depth=float(combined_results[i,2]),model=PREM)
        radius.append(r)
        W, L = width_length(ax1=r,wave=wave,fc=float(combined_results[i,1]),depth=float(combined_results[i,2]),model=PREM)
        t = ellip_tao(M0=float(combined_results[i,3]),W=W,L=L)
        tao1.append(t)

    # Second method: define length and width according to slab thickness
    tao2 = []
    depth2 = []
    mag2 = []
    for i in np.arange(len(combined_results[:,0])):
        r = slab_thk(lat=float(combined_results[i,4]),lon=float(combined_results[i,5]))
        if r > 0:
            W, L = width_length(ax1=r,wave=wave,fc=float(combined_results[i,1]),depth=float(combined_results[i,2]),model=PREM)
            t = ellip_tao(M0=float(combined_results[i,3]),W=W,L=L)
            tao2.append(t)
            depth2.append(float(combined_results[i,2]))
            mag2.append(float(combined_results[i,6]))

    tao_cir = []
    for i in np.arange(len(combined_results[:,0])):
        t = cir_tao(fc=float(combined_results[i,1]),wave=wave,depth=float(combined_results[i,2]),model=PREM,M0=float(combined_results[i,3]))
        tao_cir.append(t)

    # plot figures
    fig, ax = plt.subplots(1,1,figsize=[6.5,5])
    ax.plot(np.array(combined_results[:,6]),np.array(tao1)*1e-6,linestyle='',marker='{}'.format(marker),mec='black',mfc='cyan',markersize=8,markeredgewidth=1,alpha=0.7,label='ellip $\Delta\sigma_{}$ cir'.format(wave))
    popt,pcov = curve_fit(linear_func, np.array(combined_results[:,6]),np.log10(np.array(tao1)*1e-6))    
    print(wave,popt)
    ax.plot(np.array(mag2),np.array(tao2)*1e-6,linestyle='',marker='{}'.format(marker),mec='black',mfc='black',markersize=8,markeredgewidth=1,alpha=0.7,label='ellip $\Delta\sigma_{}$ slab'.format(wave))
    ax.plot(np.array(combined_results[:,6]),np.array(tao_cir)*1e-6,linestyle='',marker='{}'.format(marker),mec='black',mfc='{}'.format(color),markersize=8,markeredgewidth=1,alpha=0.7,label='cir $\Delta\sigma_{}$'.format(wave))

    ax.set_xlabel('Mw',fontsize=14)
    ax.set_ylabel('Stress drop (MPa)',fontsize=14)
    ax.set_title('Elliptical cracks',fontsize=16)
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig('result_{}.pdf'.format(wave))

    fig2,ax2 = plt.subplots(1,1,figsize=[6.5,5])
    ax2.plot(combined_results[:,6],radius,marker='d',linestyle='',mec='black',mfc='orange',markersize=8)
    ax2.set_xlabel('Mw',fontsize=14)
    ax2.set_ylabel('radius',fontsize=14)
    ax2.set_yscale('log')
    fig2.savefig('fc_radius.pdf')
    print(np.vstack((np.array(combined_results[:,0]),np.array(radius))))

main()
