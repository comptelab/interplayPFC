from __future__ import division
from numpy import *
import sys
from scipy.stats import zscore
sys.path.insert(0, '../../helpers/')
from pickle import dump,load
from pickle import load,dump
from scipy.stats import *
import warnings
warnings.filterwarnings("ignore")


W1=0.05
W2=0.3


f=open("../../Data/neural_data_concat.pickle","r")
data=load(f)
f.close()

f=open("../../Data/monkey_behavior_by_neuron.pickle","r")
data_d=load(f)
f.close()

f=open("../../Data/monkey_behavior_by_sessions.pickle","r")
session_close=load(f)
f.close()

nneurons = len(data)



xx=arange(-6,5-W2+W1,W1)
all_fixs=[]
counts = [[[] for _ in xx] for _ in range(nneurons)]
for n in range(nneurons):
    session = data_d[n]["session"]
    i_close_trials = session_close[session]["i_close_trials"]
    i_prev_trials = session_close[session]["i_prev_trials"] 
    fixs = array(data_d[n]["INDX"][i_prev_trials,3]) - array(data_d[n]["INDX"][i_prev_trials,4])
    idx_f = fixs > -1700 
    all_fixs.append(fixs[idx_f])
    assert len(i_close_trials) - len(idx_f) == 0
    spikes=data[n]
    for i,beg in enumerate(xx):
        end = beg + W2
        spike_count = nansum((spikes[idx_f,:] > beg) & (spikes[idx_f,:] < end),1)
        counts[n][i] = spike_count



counts=array(counts)
zsc=array([nanmean(zscore(row_stack(counts[n,:])),1) for n in range(nneurons)])
mean_fr=nanmean(zsc,0)
stderr=nanstd(zsc,0)/sqrt(nneurons)


xx-=1
fixon=round(mean(concatenate(all_fixs)/1000),2)
f=open("../preprocessed_data/prefiring_rate_fig2.pickle","w")
dump([xx,mean_fr,stderr,fixon],f)
f.close()
