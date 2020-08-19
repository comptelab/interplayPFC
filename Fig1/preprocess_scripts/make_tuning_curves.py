
from __future__ import division
import numpy as np
from math import atan2
from scipy.stats import ttest_rel,ttest_1samp,ttest_ind
from matplotlib.pylab import *
import sys
from cmath import phase,polar
from scipy.stats.mstats import zscore
from scipy.stats import circmean,circvar
import random
from scipy.stats import spearmanr as spear
from numpy import random as rd
import itertools
sys.path.insert(0, '../../helpers/')
from circ_stats import *
from numpy.fft import rfft,irfft
import scipy
from joblib import Parallel, delayed  
import multiprocessing
import scipy.signal
from pickle import dump,load
from scikits import bootstrap
import seaborn as sns
from pickle import load,dump



f=open("../../Data/neural_data_concat.pickle","r")
data=load(f)
f.close()
f=open("../../Data/monkey_behavior_by_neuron.pickle","r")
data_d=load(f)
f.close()
f=open("../../Data/monkey_behavior_by_sessions.pickle","r")
session_close=load(f)
f.close()

sns.set_context("poster", font_scale=1)
sns.set_style("ticks")


lw=4

nneurons = len(data)
W1=0.1
W2_tc = 0.3 

xx=arange(-9,5-W2_tc+W1,W1)

## FOR THE TUNING
f = open("prefs_lm.pickle")
pref = load(f)

cues_pi = arange(8)/8.0*2*pi

def f_tc(w,W,filter_fix=True):
    tc_fix = []
    for n in range(nneurons):
        prefi_p1 = argmax(abs(circdist(cues_pi,pref[n])))
        meanfr_cue = zeros([8,2])
        session = data_d[n]["session"]
        i_close_trials = session_close[session]["i_close_trials"]
        i_prev_trials = session_close[session]["i_prev_trials"] 
        spikes=data_d[n]["D"][i_close_trials,:]
        spikes=data[n]
        indx_p = array(data_d[n]["INDX"][i_prev_trials,:])
        indx_p[:,0]=indx_p[:,0]-1
        fixs = array(data_d[n]["INDX"][i_prev_trials,3]) - array(data_d[n]["INDX"][i_prev_trials,4])
        idx_f = fixs > -1700
        all_fixs.append(fixs[fixs > -1700])
        for cue in range(8):    
            t_max,t_min = (w+W, w)
            trials = (indx_p[:,0] == cue)
            if filter_fix:
                trials &= idx_f
            meanfr_cue[cue,0] = nanmean(sum((spikes[trials,:] < t_max) & (spikes[trials,:]>t_min),1))/W
        tc_fix += [concatenate([meanfr_cue[:,0][prefi_p1+1:],meanfr_cue[:,0][:prefi_p1+1]])]
    return tc_fix



TC_FIX=[]
all_fixs=[]
for w in xx:
    tc_fix=f_tc(w,W2_tc,False)
    TC_FIX.append(tc_fix)



TC_FIX2=amap(lambda x: amap(zscore,x),TC_FIX)

xx-=1

### plot tuning curves

t0=60
t1=72
t2=95
t3=85


f=open("../preprocessed_data/tuning_tcfix2.pickle","w")
dump(TC_FIX2,f)
f.close()
