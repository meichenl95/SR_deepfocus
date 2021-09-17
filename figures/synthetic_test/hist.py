#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

def gauss_function(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))

cut = np.genfromtxt('cut_fc.txt')
cut_low = cut[0,:]
cut_upp = cut[1,:]
for i in np.arange(len(cut_low)):
    if cut_low[i]==0:
        cut_low[i] = 1
    if cut_upp[i]==0:
        cut_upp[i] = 10**0.5

fig,ax = plt.subplots(1,2,figsize=[7,4])
n_low,bins_low,patches_low = ax[0].hist(np.log10(cut_low),bins=25,facecolor='green',edgecolor='black',alpha=0.75,lw=0.1,density=True)
n_upp,bins_upp,patches_upp = ax[1].hist(np.log10(cut_upp),bins=25,facecolor='green',edgecolor='black',alpha=0.75,lw=0.1,density=True)
ax[0].set_xlabel('log(Lower frequency boundary (Hz))',size=8)
ax[1].set_xlabel('log(Upper frequency boundary (Hz))',size=8)
ax[0].set_ylabel('Density',size=8)
ax[0].grid(True)
ax[0].tick_params(which='both',direction='in',grid_color='gray',grid_alpha=0.75,grid_linewidth=0.1,grid_linestyle='--',labelsize=6)
ax[1].grid(True)
ax[1].tick_params(which='both',direction='in',grid_color='gray',grid_alpha=0.75,grid_linewidth=0.1,grid_linestyle='--',labelsize=6)

(mu_low,sigma_low) = norm.fit(np.log10(cut_low))
y_low = norm.pdf(bins_low,mu_low,sigma_low)
print(mu_low,sigma_low)
ax[0].plot(bins_low,y_low,'r--',linewidth=2)
(mu_upp,sigma_upp) = norm.fit(np.log10(cut_upp))
y_upp = norm.pdf(bins_upp,mu_upp,sigma_upp)
print(mu_upp,sigma_upp)
ax[1].plot(bins_upp,y_upp,'r--',lw=2)

ax[0].set_title(r'$\mu=%.3f,\ \sigma=%.3f$' %(mu_low,sigma_low),size=10)
ax[1].set_title(r'$\mu=%.3f,\ \sigma=%.3f$' %(mu_upp,sigma_upp),size=10)

fig.tight_layout()
plt.savefig('cut.pdf')

