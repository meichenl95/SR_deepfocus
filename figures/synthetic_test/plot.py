#!/home/meichen/anaconda3/bin/python

def smooth(y,n):
    import numpy as np

    y_smooth = np.zeros(len(y))
    for i in np.arange(len(y)):
        if i < (n-1)/2 or i > (len(y)-(n-1)/2-1):
            y_smooth[i] = y[i]
        else:
            y_smooth[i] = np.sum(y[i-int((n-1)/2):i+int((n+1)/2)])/n
    return y_smooth

def func(x,a,b,c):
    import numpy as np
    return np.log10(a) + np.log10(1 + x**2/b**2) - np.log10(1 + x**2/c**2)

def fit_lower(x,y,fc_lower):
    from scipy.optimize import curve_fit
    import numpy as np

    x_cut = []
    y_cut = []
    for i in np.arange(len(x)):
        if x[i]>fc_lower:
            x_cut.append(x[i])
            y_cut.append(y[i])
    x_cut = np.array(x_cut)
    y_cut = np.array(y_cut)
    try:
        popt,pcov = curve_fit(func,x_cut,y_cut,bounds=([1,0.,0.],[100000,10.,40.]),method='trf',loss='huber',f_scale=0.1)
        return popt[0],popt[1],popt[2]
    except:
        return 0,0,0

def fit_upper(x,y,fc_upper):
    from scipy.optimize import curve_fit
    import numpy as np

    x_cut = []
    y_cut = []
    for i in np.arange(len(x)):
        if x[i]<fc_upper:
            x_cut.append(x[i])
            y_cut.append(y[i])
    x_cut = np.array(x_cut)
    y_cut = np.array(y_cut)
    try:
        popt,pcov = curve_fit(func,x_cut,y_cut,bounds=([1,0.,0.],[100000,10.,40.]),method='trf',loss='huber',f_scale=0.1)
        return popt[0],popt[1],popt[2]
    except:
        return 0,0,0

def find_range(**kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from sys import argv

    number = kwargs.get('number')
    window = kwargs.get('window')

    fc_lower = 10**np.linspace(-1.3,0,50)
    fc_upper = 10**np.linspace(-0.8,0.5,50)
    
    f = 10**np.linspace(-1.6,0.9,1001)
    amp = func(f,30,15,0.3)
    noise = np.random.normal(0,0.2,1001)
    amp_noise = amp+noise
#    fig,ax = plt.subplots(1,3,figsize=[10,3])
#    ax[2].plot(f,amp_noise,lw=0.3)
#    ax[2].set_xscale('log')
#    ax[2].set_xlabel('Frequency (Hz)')

    fcm_est = []
    fce_est = []
    mr_est = []
    cut_fc_lower = 0
    for i in np.arange(50):
        a,b,c = fit_lower(f,amp_noise,fc_lower[i])
        mr_est.append(a)
        fce_est.append(b)
        fcm_est.append(c)
    lower_smooth = smooth(fcm_est,1)
    for i in np.arange(len(lower_smooth)):
        if lower_smooth[i] > 10**(np.log10(0.3)+window) or lower_smooth[i] < 10**(np.log10(0.3)-window):
            cut_fc_lower = fc_lower[i]
            break
#    ax[0].scatter(fc_lower,fcm_est,s=2)
#    ax[0].set_xlabel('fc_lower')
#    ax[0].set_yscale('log')
#    ax[0].set_xscale('log')
#    ax[0].plot(fc_lower, lower_smooth,lw=0.3)
#    ax[0].hlines(10**(np.log10(0.3)+window),xmin=0.01,xmax=1,colors='black',lw=0.5)
#    ax[0].hlines(10**(np.log10(0.3)-window),xmin=0.01,xmax=1,colors='black',lw=0.5)

    fcm_est = []
    fce_est = []
    mr_est = []
    cut_fc_upper = 0
    for i in np.arange(50):
        a,b,c = fit_upper(f,amp_noise,fc_upper[i])
        mr_est.append(a)
        fce_est.append(b)
        fcm_est.append(c)
    upper_smooth = smooth(fcm_est,1)
    for i in np.arange(len(upper_smooth)-1,-1,-1):
        if upper_smooth[i] > 10**(np.log10(0.3)+window) or upper_smooth[i]<10**(np.log10(0.3)-window):
            cut_fc_upper = fc_upper[i]
            break
#    ax[1].scatter(fc_upper,fcm_est,s=2)
#    ax[1].set_xlabel('fc_upper')
#    ax[1].set_yscale('log')
#    ax[1].set_xscale('log')
#    ax[1].plot(fc_upper,upper_smooth,lw=0.3)
#    ax[1].hlines(10**(np.log10(0.3)+window),xmin=0.1,xmax=4,colors='black',lw=0.5)
#    ax[1].hlines(10**(np.log10(0.3)-window),xmin=0.1,xmax=4,colors='black',lw=0.5)

#    plt.savefig('cut_{}.png'.format(number))
#    plt.close()
    print(cut_fc_lower,cut_fc_upper)
    return cut_fc_lower,cut_fc_upper

def main():
    import numpy as np
    import matplotlib.pyplot as plt

    cut_fc_lower = []
    cut_fc_upper = []
    for i in np.arange(500):
        print(i)
        temp_lower,temp_upper = find_range(number=i,window=0.05)
        cut_fc_lower.append(temp_lower)
        cut_fc_upper.append(temp_upper)

    np.savetxt('cut_fc.txt',np.vstack((np.array(cut_fc_lower),np.array(cut_fc_upper))))
    fig,ax = plt.subplots(1,2,figsize=[8,4])
    ax[0].hist(np.log10(cut_fc_lower),bins=15)
    ax[1].hist(np.log10(cut_fc_upper),bins=15)
    ax[0].set_xlabel('fc_lower cut')
    ax[1].set_xlabel('fc_upper cut')
    plt.savefig('cut.pdf')

main()    
