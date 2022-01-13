from psychopy import core, data, event, visual, gui, monitors
import os
from pathlib import Path
from psychopy.hardware import keyboard
from psychopy.tools.monitorunittools import deg2pix
from scipy.ndimage import maximum_filter
import numpy as np
from numpy.fft import fftn, fftshift, fftfreq
from scipy.ndimage import rotate
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import zip_longest
import numpy.ma as ma
from scipy import sparse

os.getcwd()
os.chdir(os.getcwd())

import motionenergy as me
import stimulus as stim

#  Determine the parameters of the display
mon1 = monitors.Monitor('testMonitor')
mon1.setDistance(50) #cm
mon1.setWidth(30) #cm
mon1.setSizePix([1920, 1080])
mon1.saveMon()

## compute ppd
ppd = int(deg2pix(1, mon1, correctFlat=False)) ## for monitor= mon1
framerate = 1 / 60  # display temporal resolution

## filters
filter_shape = 64, 64, 25 
filter_res = 1 / ppd, 1 / ppd, framerate
filters = me.motion_filters(filter_shape, filter_res, csigx=0.35, k=125) ## create filters

## function to compute mean by frame
## input: xx = list
## output: list = mean of the ith elements of xx
def mean_iwise(xx):
    n_frame = [item.shape[0] for item in xx]
    mean = []
    for f in range(np.max(n_frame)):
        k = 0
        x = 0
        for i in range(len(xx)):
            if n_frame[i] > f:
                k += 1
                x += xx[i][f]
        mean.append(x/k)
    return(mean)

##  load data
# expInfo={'participant':''}
# expName="RDK"
# dlg=gui.DlgFromDict(dictionary=expInfo, title=expName)
# if dlg.OK==False: core.quit() #user pressed cancel

_thisDir =  os.path.abspath(os.getcwd())
# data_dir = _thisDir + os.sep + 'data' + os.sep + expInfo['participant']  
# data_array = np.load(data_dir + os.sep +  expInfo['participant'] + '.npy', allow_pickle=True,)
data_dir = _thisDir + os.sep +  'data' + os.sep + 'behavior' + os.sep + 'pour_thibault' 
data_array = np.load(data_dir + os.sep +   's1.npy', allow_pickle=True,)
number_of_dots =  data_array.shape[0]
dotsize = 4 ## needed to recover the true frame via max filter
# 1ere colonne: numéro de l'essai
# 2eme colonne: cohérence
# 3eme colonne force
# 4eme colonne: coordonnées
# 5eme colonne: nombre de frames perdues
# 6eme colonne: réponse ('g', 'd')
# 7eme colonne: accuracy 
data_array = data_array[np.where(data_array[:,1] == .02)] ## select coherence = .02
force = np.unique(data_array[:,2])
## compute kernel
kernel_stim = []
kernel_resp = []
for f in range(3):
    print('f = ',f)
    data_right = data_array[np.where((data_array[:,5] == 'd') & (data_array[:,2] == force[f]))]
    data_left = data_array[np.where((data_array[:,5] == 'g') & (data_array[:,2] == force[f]))]
    energy_left = []  
    energy_right = []

    for i in range(data_right.shape[0]): ##for each trial
        print('right i = ',i)
        dots = data_right[i,3] ##dots frames for trial i
        for k in range(dots.shape[0]): ## recover dots coordonnates from sparse matrices
                dots[k] = dots[k].toarray()*1            
        dots = np.dstack(dots)
        dots = maximum_filter(dots,  dotsize) ##build dots of size=dotsize
        dots_energy = me.apply_motion_energy_filters(dots, filters) ## apply filters on trial i
        energy = np.sum(dots_energy, axis=(0, 1)) ## motion energy on trial i
        energy = energy.astype(float)
        energy_right.append(energy)    
    right_mean = mean_iwise(energy_right)

    for i in range(data_left.shape[0]): ##for each trial
        print('left i = ',i)
        dots = data_left[i,3] ##dots frames for trial i
        for k in range(dots.shape[0]): ## recover dots coordonnates from sparse matrices
            dots[k] = dots[k].toarray()*1
        dots = np.dstack(dots)
        dots_energy = me.apply_motion_energy_filters(dots, filters) ## apply filters on trial i
        energy = np.sum(dots_energy, axis=(0, 1)) ## motion energy on trial i
        energy = energy.astype(float)
        energy_left.append(energy)           
    left_mean = mean_iwise(energy_left)

    n = np.min([len(right_mean), len(left_mean)])
    right_array_resp = np.array(right_mean)[-n:]
    left_array_resp = np.array(left_mean)[-n:]
    net_resp = right_array_resp - left_array_resp
    kernel_resp.append(net_resp)

    right_array_stim = np.array(right_mean)[:n]
    left_array_stim = np.array(left_mean)[:n]
    net_stim = right_array_stim - left_array_stim
    kernel_stim.append(net_stim)

np.save('kernel_resp.npy', kernel_resp, allow_pickle=True)
np.save('kernel_stim.npy', kernel_stim, allow_pickle=True)

y0 = kernel_resp[0]
y1 = kernel_resp[1]
y2 = kernel_resp[2]
x0 = np.linspace(1, y0.shape[0],  y0.shape[0]) - y0.shape[0]
x1 = np.linspace(1, y1.shape[0], y1.shape[0])  - y1.shape[0]
x2 = np.linspace(1, y2.shape[0],  y2.shape[0])  - y2.shape[0]
plt.plot(x0,y0,label='force =' +  str(force[0]))
plt.plot(x1,y1, label='force =' +  str(force[1]))
plt.plot(x2,y2, label='force =' +  str(force[2]))
plt.title(label="response aligned kernel")
ax.set(xlabel="time",
       ylabel="sensory weights (a.u.)")
plt.legend()
plt.savefig('resp_aligned.png')
plt.close('all')

y0 = kernel_stim[0]
y1 = kernel_stim[1]
y2 = kernel_stim[2]
x0 = np.linspace(1, y0.shape[0],  y0.shape[0])
x1 = np.linspace(1, y1.shape[0], y1.shape[0])
x2 = np.linspace(1, y2.shape[0],  y2.shape[0])
plt.plot(x0,y0,label='force =' +  str(force[0]))
plt.plot(x1,y1, label='force =' +  str(force[1]))
plt.plot(x2,y2, label='force =' +  str(force[2]))
plt.title(label="stimulus aligned kernel")
ax.set(xlabel="time",
       ylabel="sensory weights (a.u.)")
plt.legend()
plt.savefig('stim_aligned.png')
plt.close('all')
    
