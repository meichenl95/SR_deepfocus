#!/home/meichen/anaconda3/bin/python
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
import glob

def func_Boatwright(x,a,b,c):
    return np.log10(a) + 1./2.* np.log10(1+x**4/b**4) - 1./2.*np.log10(1+x**4/c**4)

def bootstrap(filename):
    data = np.genfromtxt('{}'.format(filename))
    l = len(data[:,1])
    c_list = []
    
    [a,b,c],pcov = curve_fit(func_Boatwright,data[:,0],np.log10(data[:,1]),bounds=([1,0.025,0.025],[100000,4.,4.]),method='trf',loss='huber',f_scale=0.1)
    
    residual = np.log10(data[:,1]) - func_Boatwright(data[:,0],a,b,c)
    fig,ax = plt.subplots(1,1,figsize=[5,2])
    for i in np.arange(1000):
        random_index = np.random.randint(0,l,size=l)
        new_ydata = np.log10(data[:,1]) + [residual[j] for j in random_index]
        try:
            [new_a,new_b,new_c], pcov = curve_fit(func_Boatwright,data[:,0],new_ydata,bounds=([1,0.025,0.025],[100000,4.,4.]),method='trf',loss='huber',f_scale=0.1)
            c_list.append(new_c)
            ax.plot(np.log10(data[:,0]),func_Boatwright(data[:,0],new_a,new_b,new_c))
        except RuntimeError:
            print("Error - curve_fit failed")
    
    ax.plot(np.log10(data[:,0]),np.log10(data[:,1]))
    fig.savefig('{}.png'.format(filename))
    plt.close()
    
    std = np.std(np.array(c_list),ddof=1)
    mean = np.mean(np.array(c_list))
    
def main(f):
    filelist = glob.glob('{}'.format(f))
    for filename in filelist:
        print(filename)
        filename = filename.split('/')[-1]
        bootstrap(filename)

main('*.sr')
