from __future__ import division
from numpy import *
import numpy as np
import sys
sys.path.insert(0, '../../helpers/')
from circ_stats import *
import random
import itertools
import scipy
from joblib import Parallel, delayed  
import multiprocessing
import scipy.signal
from scipy.signal import correlate
from pickle import load,dump
from matplotlib.pylab import amap,shuffle
import warnings
warnings.filterwarnings("ignore")

w1=0.01 
w2=0.01 

W1=0.5 
W1=1
W2=0.05 

wj = 0.05 
n_wj = int(wj / w1)


dist = 60
slack = 40

cues = range(8)

n_perms = 1000 

f=open("../../Data/neural_data_concat.pickle","r")
data_c=load(f)
f.close()

f=open("../../Data/monkey_behavior_by_neuron.pickle","r")
data=load(f)
f.close()

f=open("../../Data/monkey_behavior_by_sessions.pickle","r")
session_close=load(f)
f.close()

prefs=loadtxt("prefs.txt")

nneurons = len(data)


time=arange(-5,2.5,W2)


def xcorrelation1(p1,p2,s="full"):
    
    def corr(p1,p2,n_trials):
        return [correlate(p1[i]-mean(p1[i]), p2[i]-mean(p2[i]),s) for i in range(n_trials)]
    
    def jitter(p1,n_wj):
        for trial in arange(0,len(p1)):
            for beg in arange(0,len(p1[trial]),n_wj):
                end = beg + n_wj
                shuffle(p1[trial][beg:end])
        return p1
    
    n_trials = len(p1)

    acor = corr(p1,p2,n_trials)

    shuff=[]
    
    for _ in range(n_perms):
        p1=jitter(p1,n_wj)
        shuff.append(corr(p1,p2,n_trials))
        
    acor = acor - mean(shuff,0)
    return acor



pairs={}
for i in range(nneurons):
    s=data[i]["session"]
    if s in pairs.keys():
        pairs[s].add(i)
    else:
        pairs[s]=set([i])

dist_pref=[]
pairs = map(list,pairs.values())
mono_syn2=[]
for pair in pairs:
    for n1,n2 in itertools.combinations(pair,2):
        dist_p=circdist(prefs[n1],prefs[n2])[0]
        mn=circmean([prefs[n1],prefs[n2]])
        if ~isnan(dist_p): # we have a pref for both neurons
            dist_pref.append(dist_p)
            mono_syn2.append([n1,n2])



# REMOVE WEIRD PAIR 
del dist_pref[158]
del mono_syn2[158]

mono_syn2 = array(mono_syn2)[np.abs(dist_pref)<radians(dist)]

prev=[]

## SELECT WHICH TRIALS
trials_on_pref = []
trials_out_pref = []
for i,(p1,p2) in enumerate(mono_syn2):
    
    if circdist(prefs[p1],prefs[p2])>0:
        p1,p2 = p2,p1
    
    assert data[p1]["session"] == data[p2]["session"]
    
    session = data[p1]["session"]
    
    i_close_trials = session_close[session]["i_close_trials"]
    i_prev_trials = session_close[session]["i_prev_trials"] 
    
    
    indx_p = array(data[p1]["INDX"][i_prev_trials,:])
    indx_p[:,0]=indx_p[:,0]-1
    
    rad_previous = indx_p[:,0]/8.0*(2*pi)
    
    idx_a = abs(circdist(prefs[p1],rad_previous))<abs(circdist(prefs[p1],prefs[p2]))+radians(slack)
    idx_b = abs(circdist(prefs[p2],rad_previous))<abs(circdist(prefs[p1],prefs[p2]))+radians(slack)
    pref_dir=idx_a & idx_b
    
    trials_on_pref.append(pref_dir )
    trials_out_pref.append(~pref_dir )

    prev.append(rad_previous)


## SPIKE COUNT AND CALL XCORR FOR 1 PAIR, 1 TIME
def corr_par(beg_,pair):

    end_ = beg_ + W1
    print "correlating pair: %i, time: [%.2f,%.2f]" % (pair,beg_,end_)
    xx=arange(beg_,end_-w2+w1,w1).round(2)

    p1,p2 = mono_syn2[pair] 
    if circdist(prefs[p1],prefs[p2])>0:
        p1,p2 = p2,p1

    spikes1=data_c[p1]
    spikes2=data_c[p2]
    
    spikes_pr1 = [] 
    spikes_ant1 = []
    
    spikes_pr2 = [] 
    spikes_ant2 = []
    
    pref_dir = trials_on_pref[pair]
    ant_dir = trials_out_pref[pair]


    # spike counts    
    for j,beg in enumerate(xx):
        end = beg + w2
        spike_count1_pref = nansum((spikes1[pref_dir] <= end) & (spikes1[pref_dir] > beg),1)
        spike_count2_pref = nansum((spikes2[pref_dir] <= end) & (spikes2[pref_dir] > beg),1)
        
        spike_count1_ant = nansum((spikes1[ant_dir] <= end) & (spikes1[ant_dir] > beg),1)
        spike_count2_ant = nansum((spikes2[ant_dir] <= end) & (spikes2[ant_dir] > beg),1)
        
        spikes_pr1.append(spike_count1_pref)
        spikes_pr2.append(spike_count2_pref)
        
        spikes_ant1.append(spike_count1_ant)
        spikes_ant2.append(spike_count2_ant)
    

    
    spikes_pr1= array(spikes_pr1).T
    spikes_pr2= array(spikes_pr2).T
    spikes_ant1= array(spikes_ant1).T
    spikes_ant2= array(spikes_ant2).T

    
    corr_on_pref = xcorrelation1(spikes_pr1,spikes_pr2)
    corr_out_pref = xcorrelation1(spikes_ant1,spikes_ant2)

    return corr_on_pref,corr_out_pref,[spikes_pr1,spikes_pr2],[spikes_ant1,spikes_ant2]


num_cores = multiprocessing.cpu_count()

pair = int(sys.argv[1])
res = Parallel(n_jobs=num_cores)(delayed(corr_par)(beg,pair) for beg in time)

np.savez_compressed("xcorr_pairs/pair_%i_%.2f" % (pair,W1), [res,time])
