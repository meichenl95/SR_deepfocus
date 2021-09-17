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
    
    fig,ax = plt.subplots(3,1,figsize=[6,7])

    # seismograms
    time = np.linspace(-10,140,master.stats.sac['npts'])
    ax[0].plot(time,master.data/(np.max(master.data)-np.min(master.data)),lw=1,color='black',alpha=0.8,label='master event')
    ax[0].plot(time,egf.data/(np.max(egf.data)-np.min(egf.data)),lw=1,color='blue',alpha=0.6,label='eGf')
    ax[0].arrow(-5,-0.3,40,0,lw=0.4,length_includes_head=True,head_width=0.03,head_length=2,color='red',alpha=0.8)
    ax[0].arrow(15,-0.35,40,0,lw=0.4,length_includes_head=True,head_width=0.03,head_length=2,color='red',alpha=0.8)
    ax[0].arrow(35,-0.4,40,0,lw=0.4,length_includes_head=True,head_width=0.03,head_length=2,color='red',alpha=0.8)
    ax[0].arrow(55,-0.45,40,0,lw=0.4,length_includes_head=True,head_width=0.03,head_length=2,color='red',alpha=0.8)
    ax[0].arrow(75,-0.5,40,0,lw=0.4,length_includes_head=True,head_width=0.03,head_length=2,color='red',alpha=0.8)
    ax[0].plot([-5,-5],[-0.28,-0.32],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([35,35],[-0.28,-0.32],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([15,15],[-0.33,-0.37],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([55,55],[-0.33,-0.37],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([35,35],[-0.38,-0.42],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([75,75],[-0.38,-0.42],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([55,55],[-0.43,-0.47],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([95,95],[-0.43,-0.47],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([75,75],[-0.48,-0.52],lw=0.4,color='r',alpha=0.8)
    ax[0].plot([115,115],[-0.48,-0.52],lw=0.4,color='r',alpha=0.8)
    ax[0].text(0.05,0.9,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax[0].transAxes)
    ax[0].set_yticks([-0.5,0,0.5])
    ax[0].set_yticklabels([-0.5,0,0.5],fontsize=8)
    ax[0].tick_params(axis='x',labelsize=8)
    ax[0].legend()
    ax[0].set_xlabel('Time (s)',size=10)
    ax[0].set_ylabel('Normalized seismograms',size=10)

    # spectra
    m1 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np1.master')
    m2 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np2.master')
    m3 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np3.master')
    m4 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np4.master')
    m5 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np5.master')
    e1 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np1.egf')
    e2 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np2.egf')
    e3 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np3.egf')
    e4 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np4.egf')
    e5 = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.Np5.egf')
    m = (m1+m2+m3+m4+m5)/5.
    e = (e1+e2+e3+e4+e5)/5.
    lns1 = ax[1].loglog(m[:,0],m[:,1],lw=1,color='black',label='master event')
    lns2 = ax[1].loglog(e[:,0],e[:,1],lw=1,color='blue',label='eGf')
    ax[1].set_ylabel('Spectral amplitude',size=10)
    sr = np.genfromtxt('BK.MOD..BHZ.P.-5.40.5.sr')
    ax_twin = ax[1].twinx()
    lns3 = ax_twin.loglog(sr[:,0],sr[:,1],lw=1.5,color='red',alpha=0.3,label='Spectral ratio')
    ax_twin.set_ylabel('Ratios',size=10,rotation=270)
    ax_twin.set_yticks([1,10])
    ax_twin.set_yticklabels([1,10],fontsize=8)
    ax[1].text(0.05,0.9,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax[1].transAxes)
    ax[1].set_xticks([0.1,1])
    ax[1].set_xticklabels([0.1,1],fontsize=8)
    ax[1].set_yticks([0.01,0.1,1,10])
    ax[1].set_yticklabels([0.01,0.1,1,10],fontsize=8)
    ax[1].minorticks_off()

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns,labs)

    # multi eGfs
    egf_2372840 = np.genfromtxt('all.P.85.egf_2372840.stn.138.Np.sr')
    egf_1547900 = np.genfromtxt('all.P.85.egf_1547900.stn.43.Np.sr')
    egf_1327033 = np.genfromtxt('all.P.85.egf_1327033.stn.54.Np.sr')
    ax[2].loglog(egf_2372840[:,0],egf_2372840[:,1],lw=1,color='black',label='eGf 1',alpha=0.8)
    ax[2].loglog(egf_2372840[:,0],func(egf_2372840[:,0],4.1136,0.505,0.288),lw=1.5,ls='--',color='grey',alpha=0.5)
    ax[2].loglog(0.288,func(0.288,4.1136,0.505,0.288),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='black',markeredgewidth=0.1)
    ax[2].loglog(egf_1547900[:,0],egf_1547900[:,1],lw=1,color='orange',label='eGf 2',alpha=0.8)
    ax[2].loglog(egf_1547900[:,0],func(egf_1547900[:,0],2.7345,0.5589,0.407),lw=1.5,ls='--',color='grey',alpha=0.5)
    ax[2].loglog(0.407,func(0.407,2.7345,0.5589,0.407),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='orange',markeredgewidth=0.1)
    ax[2].loglog(egf_1327033[:,0],egf_1327033[:,1],lw=1,color='magenta',label='eGf 3',alpha=0.8)
    ax[2].loglog(egf_1327033[:,0],func(egf_1327033[:,0],3.58,0.6466,0.3772),lw=1.5,ls='--',color='gray',alpha=0.5)
    ax[2].loglog(0.3772,func(0.3772,3.58,0.6466,0.3772),marker='v',markeredgecolor='black',markersize=5,markerfacecolor='magenta',markeredgewidth=0.1)
    ax[2].text(0.05,0.9,'(c)',horizontalalignment='center',verticalalignment='center',transform=ax[2].transAxes)
    ax[2].set_xlabel('Frequency (Hz)',size=10)
    ax[2].set_ylabel('Stacked spectral ratios',size=10)    
    ax[2].minorticks_off()
    ax[2].set_xticks([0.1,1])
    ax[2].set_xticklabels([0.1,1],fontsize=8)
    ax[2].set_yticks([1,2,3,4,6])
    ax[2].set_yticklabels([1,2,3,4,6],fontsize=8)
    ax[2].legend()
    
    plt.tight_layout()
    plt.savefig('figure2.pdf')
main()
