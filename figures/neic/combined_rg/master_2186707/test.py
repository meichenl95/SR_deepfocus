#!/home/meichen/anaconda3/bin/python

def func(x,a,b,c):
    return a * (1+(x/b)**2)/(1+(x/c)**2)

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    import obspy
    from matplotlib.ticker import ScalarFormatter

    master = obspy.read('*.master.cut')[0]
    egf = obspy.read('*.egf.cut')[0]

    fig,ax = plt.subplots(1,1,figsize=[6,2.33])

    # multi eGfs
    egf_2372840 = np.genfromtxt('all.P.85.egf_2372840.stn.138.Np.sr')
    egf_1547900 = np.genfromtxt('all.P.85.egf_1547900.stn.43.Np.sr')
    egf_1327033 = np.genfromtxt('all.P.85.egf_1327033.stn.54.Np.sr')
    ax.plot(egf_2372840[:,0],egf_2372840[:,1],lw=1,color='black',label='eGf 1',alpha=0.8)
#    ax[2].loglog(egf_2372840[:,0],func(egf_2372840[:,0],4.1136,0.505,0.288),lw=1.5,ls='--',color='grey',alpha=0.5)
#    ax[2].loglog(0.288,func(0.288,4.1136,0.505,0.288),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='black',markeredgewidth=0.1)
#    ax[2].loglog(egf_1547900[:,0],egf_1547900[:,1],lw=1,color='orange',label='eGf 2',alpha=0.8)
#    ax[2].loglog(egf_1547900[:,0],func(egf_1547900[:,0],2.7345,0.5589,0.407),lw=1.5,ls='--',color='grey',alpha=0.5)
#    ax[2].loglog(0.407,func(0.407,2.7345,0.5589,0.407),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='orange',markeredgewidth=0.1)
#    ax[2].loglog(egf_1327033[:,0],egf_1327033[:,1],lw=1,color='magenta',label='eGf 3',alpha=0.8)
#    ax[2].loglog(egf_1327033[:,0],func(egf_1327033[:,0],3.58,0.6466,0.3772),lw=1.5,ls='--',color='gray',alpha=0.5)
#    ax[2].loglog(0.3772,func(0.3772,3.58,0.6466,0.3772),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='magenta',markeredgewidth=0.1)
#    ax[2].text(0.05,0.9,'(c)',horizontalalignment='center',verticalalignment='center',transform=ax[2].transAxes)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xlabel('Frequency (Hz)',size=10)
    ax.set_ylabel('Stacked spectral ratios',size=10)
    ax.set_xticks([0.2,1,2])
    ax.set_xticklabels(["0.02!","1!",2],fontsize=8)
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(["1!","1!","3!"],fontsize=8)
#    ax[2].legend()

    plt.tight_layout()
    plt.savefig('test.pdf')

main()
