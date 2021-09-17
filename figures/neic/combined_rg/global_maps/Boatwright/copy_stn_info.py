#!/home/meichen/anaconda3/bin/python

import numpy as np
import glob
import os
import subprocess
import pandas as pd

def main():
    # S
    data = pd.read_csv('pairsfile_rgs_select.csv')
    data_array = np.array(data)
    jpath = '/home/meichen/work1/SR_Attn/pair_events'

    for i in np.arange(len(data_array[:,0])):
        subprocess.call(["saclst stlo stla f {}/master_{}/egf_{}/S/gcarc_30/*.SAC.master | gawk '{{print $2,$3}}' > {}_{}_S_30.txt".format(jpath,data_array[i,0],data_array[i,6],data_array[i,0],data_array[i,6])],shell=True)
        subprocess.call(["saclst stlo stla f {}/master_{}/egf_{}/S/gcarc_30_85/*.SAC.master | gawk '{{print $2,$3}}' > {}_{}_S_30_85.txt".format(jpath,data_array[i,0],data_array[i,6],data_array[i,0],data_array[i,6])],shell=True)

    # P
    data = pd.read_csv('pairsfile_rgp_select.csv')
    data_array = np.array(data)
    jpath = '/home/meichen/work1/SR_Attn/pair_events'

    for i in np.arange(len(data_array[:,0])):
        subprocess.call(["saclst stlo stla f {}/master_{}/egf_{}/P/gcarc_30/*.SAC.master | gawk '{{print $2,$3}}' > {}_{}_P_30.txt".format(jpath,data_array[i,0],data_array[i,6],data_array[i,0],data_array[i,6])],shell=True)
        subprocess.call(["saclst stlo stla f {}/master_{}/egf_{}/P/gcarc_30_85/*.SAC.master | gawk '{{print $2,$3}}' > {}_{}_P_30_85.txt".format(jpath,data_array[i,0],data_array[i,6],data_array[i,0],data_array[i,6])],shell=True)

main()
