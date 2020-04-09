from __future__ import division
import numpy as np
from scipy.stats import sem
from matplotlib.pylab import *
import sys
import random
import scipy
from scipy.signal import gaussian
from scipy.ndimage import filters
from joblib import Parallel, delayed  
import multiprocessing
from scikits import bootstrap
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings("ignore")
num_cores = multiprocessing.cpu_count()

sns.set_context("poster", font_scale=1.1)
sns.set_style("ticks")


W1=1
W2=0.05 

w1=0.01 
w2=0.01 


time=arange(-5,2.5-W1+W2,W2)



def sig_bar(sigs,axis,y,color):
    w=np.diff(axis)[0]
    continuity = np.diff(sigs)
    for i,c in enumerate(continuity):
        beg =axis[sigs[i]]
        end = beg+w
        fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

b = gaussian(5,10)
b = ones(5)
num_cores = multiprocessing.cpu_count()

def smooth_i(i,h):
    m=nanmean(h[:,i],1)
    p=filters.convolve1d(m, b/b.sum())
    return p


def exact_mc_perm_test2(xs, ys, nmc):
    k=0
    diff = np.mean(xs -ys)
    for j in range(nmc):
        shuffle(xs)
        k += diff < np.mean(xs -ys)
    return k/float(nmc)
        
def exact_mc_perm_test(xs, ys, nmc):
    n, m, k = len(xs), len(ys), 0
    diff = np.mean(xs,0) - np.mean(ys,0)
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        idx=np.random.permutation(n+m)
        zs=zs[idx,]
        #np.random.shuffle(zs)
        k += diff < np.mean(zs[:n,],0) - np.mean(zs[n:,],0)
    return k/float(nmc)



# load firing rate (window of 0.25 s)
files = glob(sys.argv[1]+"/*0.25*npz")
idx = [int(f.split("pair_")[1].split("_")[0]) for f in files]
files = array(files)[argsort(idx)]

res = []
for file in files:
    data,time = np.load(file, allow_pickle=True)["arr_0"]
    data_no_spikes = data 
    res.append(data_no_spikes)

res = array(res)
spikes_anti = array([[mean(sum(n[t][-1],-1)) for n in res] for t in range(len(time))])
spikes_on = array([[mean(sum(n[t][-2],-1)) for n in res] for t in range(len(time))])
spikes_diff = array([[mean(sum(n[t][-2],-1)) - mean(sum(n[t][-1],-1)) for n in res] for t in range(len(time))])

# load correlations rate (window of 1 s)
files = glob(sys.argv[1]+"/*1.00*npz")
idx = [int(f.split("pair_")[1].split("_")[0]) for f in files]
files = array(files)[argsort(idx)]
res = []
for file in files:
    data,time = np.load(file, allow_pickle=True)["arr_0"]
    data_no_spikes = data 
    res.append(data_no_spikes)

res = array(res)

## PLOTTING
x_axis_len = shape(res[0][0][0])[1]
xxx = linspace(-x_axis_len/2*w1,x_axis_len/2*w1,x_axis_len)
zero = int(floor(x_axis_len/2))
wd=1


out_pref=array([[mean(mean(r[t][1],0)[zero-wd:zero+wd+1]) for t in range(len(time))] for r in res])
on_pref=array([[mean(mean(r[t][0],0)[zero-wd:zero+wd+1]) for t in range(len(time))] for r in res])


on_pref_full = array([[mean(r[t][0],0) for t in range(len(time))] for r in res])

out_pref_full = array([[mean(r[t][1],0) for t in range(len(time))] for r in res])


# Select pairs with POS/NEG interaction
# by averaging their peak  durin whole time
pos_idx = (mean(on_pref,1)>0) & ( mean(out_pref,1)>0)
neg_idx = ( mean(on_pref,1)<0) & ( mean(out_pref,1)<0)


diff = (array(on_pref)[pos_idx]-array(out_pref)[pos_idx]).T
ci_diff_pos = array([bootstrap.ci(d) for d in diff])
p_diff_pos=exact_mc_perm_test(array(on_pref)[pos_idx],array(out_pref)[pos_idx],1000)

## 95% CI
pos_pref = (array(on_pref)[pos_idx]).T
ci_pos_pref = array([bootstrap.ci(d) for d in pos_pref])

##  SEM
sem_pos_pref=nanstd(pos_pref,1)/sqrt(sum(pos_idx))
ci_pos_pref=array([mean(pos_pref,1)+sem_pos_pref, mean(pos_pref,1)-sem_pos_pref]).T



# smooth each bootstrap, instead of smothing the bootstrapped mean - which would not make sense
h_p=array(on_pref)[pos_idx]-array(out_pref)[pos_idx]
h_p=h_p.T
idx_p=bootstrap.bootstrap_indexes(h_p[0])
res_s_p = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,h_p) for i in idx_p)

ci_h_p = array(res_s_p)
high = amap(lambda x: percentile(x,100-16),ci_h_p.T)
low = amap(lambda x: percentile(x,16),ci_h_p.T)

high_95 = amap(lambda x: percentile(x, 5),ci_h_p.T)
low_95 = amap(lambda x: percentile(x,95),ci_h_p.T)


h=array(on_pref)[neg_idx]-array(out_pref)[neg_idx]
h=h.T
idx_p=bootstrap.bootstrap_indexes(h[0])
res_s_n = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,h) for i in idx_p)

ci_h = array(res_s_n)
high_i = amap(lambda x: percentile(x,100-16),ci_h.T)
low_i = amap(lambda x: percentile(x,16),ci_h.T)

high_i_95 = amap(lambda x: percentile(x,5),ci_h.T)
low_i_95 = amap(lambda x: percentile(x,95),ci_h.T)


time = amap(lambda x: round(x,2),time)

# align to stimulus and middle of the window (0.5 s)
time=time-1+W1/2
fixoff = -7.1+1+3

figure(figsize=(10,5))
subplot(1,2,1)
fill_between(time,low,high,alpha=0.25,color=sns.xkcd_rgb["orange"],label="exc")
plot(time,mean(res_s_p,0),color=sns.xkcd_rgb["orange"],lw=3)
fill_between(time,low_i,high_i,alpha=0.25,color=sns.xkcd_rgb["greenish"],label="inh")
plot(time,mean(res_s_n,0),color=sns.xkcd_rgb["greenish"],lw=3)

plot(time,zeros(len(time)),"k--",alpha=0.3)
fill_between([0,0.5],-1,1,color="gray",alpha=0.2)
fill_between([fixoff,fixoff+.35],-1,1,color="gray",alpha=0.2)

sig_bar(find(low_i_95<0),time,[0.1*0.95,0.1],sns.xkcd_rgb["greenish"])
sig_bar(find(high_95>0),time,[0.1*0.90,0.1*0.95],sns.xkcd_rgb["orange"])


ylabel("CCSI (sp/s)$^2$")

ylim(-.1,.1)
yticks([-0.1,0,.1])

xlim(-4,0.5)

tick_params(direction='in')
l = legend(frameon=False,loc="lower right")
l.get_texts()[0].set_color(sns.xkcd_rgb["orange"])
l.get_texts()[1].set_color(sns.xkcd_rgb["greenish"])
l.legendHandles[0].set_visible(False); #hide the marker
l.legendHandles[1].set_visible(False);


subplot(1,2,2)
title("rate tuning")
mean_e = mean((spikes_diff[:,pos_idx]),1)*4
mean_i = mean((spikes_diff[:,neg_idx]),1)*4
std_e = 2*sem((spikes_diff[:,pos_idx])*4,1)
std_i = 2*sem((spikes_diff[:,neg_idx])*4,1)

low_e_t = mean_e - std_e
high_e_t = mean_e + std_e

low_i_t = mean_i - std_i
high_i_t = mean_i + std_i

fill_between(time,low_e_t,high_e_t,color=sns.xkcd_rgb["orange"], alpha=0.25)
plot(time,mean_e,color=sns.xkcd_rgb["orange"])

fill_between(time,low_i_t,high_i_t,color=sns.xkcd_rgb["greenish"], alpha=0.25)
plot(time+0.25,mean_i,color=sns.xkcd_rgb["greenish"])

plot(time,zeros(len(time)),"k--")
fill_between([0,0.5],-1,7,color="gray",alpha=0.2)
ylabel("sp/s")
xlabel("time (s)")

tick_params(direction='in')
legend(frameon=False,loc="lower right")
xlim(-4.5,1)
tight_layout()
sns.despine()


show()
