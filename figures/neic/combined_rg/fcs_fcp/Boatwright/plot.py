#!/home/meichen/anaconda3/bin/python

def select_fcsp(**kwargs):

##-------------------------------##

# This function select events with both P and S wave corner frequency

##-------------------------------##

##parameters
# dirname		the directory contains files and to save output file
# file_fcs		the name of file contains S wave corner frequency
# file_fcp		the name of file contains P wave corner frequency
# ofile			the name of output file

    import numpy as np
    import os
    import matplotlib.pyplot as plt

    dirname = kwargs.get('dirname')
    file_fcs = kwargs.get('file_fcs')
    file_fcp = kwargs.get('file_fcp')
    ofile = kwargs.get('ofile')

    fcs = np.genfromtxt('{}'.format(file_fcs))
    fcp = np.genfromtxt('{}'.format(file_fcp))

    fc_list = []
    for i in np.arange(len(fcs[:,0])):
        if fcs[i,0] in list(fcp[:,0]):
            index = list(fcp[:,0]).index(fcs[i,0])
            fc_list.append([fcs[i,1],fcp[index,1]])
            

    fc_list = np.array(fc_list)
    fig,ax = plt.subplots(1,1,figsize=[6,6])
    ax.scatter(fc_list[:,0],fc_list[:,1],marker='o',s=5,color='black')
    ax.set_xlim([0,2])
    ax.set_ylim([0,2])
    ax.set_xlabel('S wave corner frequency (Hz)')
    ax.set_ylabel('P wave corner frequency (Hz)')
    fig.savefig('{}'.format(ofile))

def main():
    
    path = '/home/meichen/Research/SR_Attn/pair_events/figures/neic/combined_rg/fcs_fcp'
    select_fcsp(dirname=path,file_fcs='fcs.txt',file_fcp='fcp.txt',ofile='fcsp.pdf')

main()
