#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt

S_magdif = np.genfromtxt('S_magdif.txt')
P_magdif = np.genfromtxt('P_magdif.txt')

fig,ax = plt.subplots(1,1,figsize=[5,5])
ax.plot(P_magdif[:,0],P_magdif[:,1],linestyle='',marker='o',markersize=7,mec='black',markeredgewidth=1,mfc='magenta',alpha=0.75,label='P')
ax.plot(S_magdif[:,0],S_magdif[:,1],linestyle='',marker='d',markersize=7,mec='black',markeredgewidth=1,mfc='blue',alpha=0.75,label='S')
#ax.plot([-0.5,2.5],[0,3],linestyle='--',color='gray',linewidth=1)
#ax.plot([0,3],[-0.5,2.5],linestyle='--',color='gray',linewidth=1)

ax.set_xlim([0,2.5])
ax.set_ylim([0,2.5])
ax.set_xlabel(r'$\Delta$M$_{\rm W}$$^{\rm Cat}$',fontsize=14)
ax.set_ylabel(r'$\Delta$M$_{\rm W}$$^{\rm Fit}$',fontsize=14)
ax.fill_between([0,2.5],[-0.5,2.0],[0.5,3.0],color='gray',alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig('magdif.pdf')
fig.savefig('magdif.png')
