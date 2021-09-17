#!/home/meichen/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import subprocess

def main():
    data = pd.read_csv('pairsfile_rgp_select.csv',skipinitialspace=True)
    data_array = np.array(data)
    jpath = '/home/meichen/work1/SR_Attn/pair_events'

    for i in np.arange(len(data_array[:,0])):
        filename = glob.glob('{}/master_{}/egf_{}/P/gcarc_85/all.*'.format(jpath,data_array[i,0],data_array[i,6]))[0].split('/')[-1]
        subprocess.call(['cp {}/master_{}/egf_{}/P/gcarc_85/{} {}_{}.{}'.format(jpath,data_array[i,0],data_array[i,6],filename,data_array[i,0],data_array[i,6],filename)],shell=True)

main()
