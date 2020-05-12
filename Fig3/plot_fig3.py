from __future__ import division
import pandas as pd
from matplotlib.mlab import find
from numpy import *
from matplotlib.mlab import *
from matplotlib.pylab import *
from scipy.io import savemat
import sys
from scipy.stats import *
from scikits import bootstrap
from scipy.stats import spearmanr
import seaborn as sns
from joblib import Parallel, delayed  
import multiprocessing
from scipy.signal import gaussian
from scipy.ndimage import filters
import statsmodels.formula.api as smf
from pickle import load
import bz2

set_printoptions(precision=4)
sns.set_context("talk", font_scale=1.3)
sns.set_style("ticks")

font = { 'style': 'italic',
        'weight': 'bold',
        'fontsize': 20
        }


b = ones(5)
num_cores = multiprocessing.cpu_count()

n_perms = 10000

# aligned to fixation on (+1) + delay (+3)
fixoff = -7.1+1+3


time_silent=65
time_react = 86

W1=1

def adjust_spines(ax, spines):  ### aesthetics, offset axies
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 20))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])

def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	continuity = diff(sigs)
	for i,c in enumerate(continuity):
		beg =axis[sigs[i]]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

def smooth_i(i,h):
    m=nanmean(h[:,i],1)
    p=filters.convolve1d(m, b/b.sum())
    return p


def amap(func, data):
	return array(list(map(func, data)))



fig=figure(figsize=(13,18))


## INTERACTION, INH VS EXC PAIRS / SCATTER PLOT OF ALL PAIRS
subplot(2,2,1)
f=open("preprocessed_data/xcorr_time_1.00.pickle")
[time,on_pref,out_pref,pos_idx,neg_idx] = load(f)

dist_elect = loadtxt("preprocessed_data/dist_elect.txt")

# select only pairs that do not come from the same electrode
#pos_idx = pos_idx & (dist_elect > 0)
#neg_idx = neg_idx & (dist_elect > 0)

ccsi=on_pref-out_pref

errorbar(array([0, 1])-0.03, [mean(ccsi[pos_idx,time_silent]), mean(ccsi[pos_idx,time_react])], yerr=[sem(ccsi[pos_idx,time_silent]), sem(ccsi[pos_idx,time_react])], color=sns.xkcd_rgb["orange"])
errorbar(array([0, 1])+0.03, [mean(ccsi[neg_idx,time_silent]), mean(ccsi[neg_idx,time_react])], yerr=[sem(ccsi[neg_idx,time_silent]), sem(ccsi[neg_idx,time_react])], color=sns.xkcd_rgb["greenish"])
errorbar(array([0, 1]), [mean(ccsi[:,time_silent]), mean(ccsi[:,time_react])], yerr=[sem(ccsi[:,time_silent]), sem(ccsi[:,time_react])], color="gray")

plot(array([0,1])-0.03,[mean(ccsi[pos_idx,time_silent]), mean(ccsi[pos_idx,time_react])],".",color=sns.xkcd_rgb["orange"],ms=15)
plot(array([0,1])+0.03,[mean(ccsi[neg_idx,time_silent]), mean(ccsi[neg_idx,time_react])],".",color=sns.xkcd_rgb["greenish"],ms=15)
plot(array([0,1]),[mean(ccsi[:,time_silent]), mean(ccsi[:,time_react])],".",color="gray",ms=15)

text(0,-0.1,"exc (n=%i)" % (sum(pos_idx)),color=sns.xkcd_rgb["orange"],fontsize=18)
text(0,-0.12,"inh (n=%i)" % (sum(neg_idx)),color=sns.xkcd_rgb["greenish"],fontsize=18)
text(0,-0.14,"all (n=%i)" % (len(pos_idx)),color="gray",fontsize=18)

xticks([0, 1],['activity-silent', 'reactivation'])
ylabel("CCSI (sps/s)$^2$", color="k")
tick_params(axis="y",direction='in')
plot([-0.25,1.25],[0,0],"k--")
xlim([-0.25,1.25])
ylim(-0.2,0.1)

# X-CORRELATION DIFFERENCE PREF VS ANTI-PREF
subplot(2,2,2)

h_p=array(on_pref)[pos_idx]-array(out_pref)[pos_idx]
h_p=h_p.T

idx_p=bootstrap.bootstrap_indexes(h_p[0])
res_s_p = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,h_p) for i in idx_p)

ci_h_p = array(res_s_p)
high = amap(lambda x: percentile(x,100-16),ci_h_p.T)
low = amap(lambda x: percentile(x,16),ci_h_p.T)

h=array(on_pref)[neg_idx]-array(out_pref)[neg_idx]
h=h.T
idx_p=bootstrap.bootstrap_indexes(h[0])
res_s_n = Parallel(n_jobs=num_cores)(delayed(smooth_i)(i,h) for i in idx_p)

ci_h = array(res_s_n)
high_i = amap(lambda x: percentile(x,100-16),ci_h.T)
low_i = amap(lambda x: percentile(x,16),ci_h.T)


fill_between(time,low,high,alpha=0.25,color=sns.xkcd_rgb["orange"],label="exc")
plot(time,mean(res_s_p,0),color=sns.xkcd_rgb["orange"],lw=3)
fill_between(time,low_i,high_i,alpha=0.25,color=sns.xkcd_rgb["greenish"],label="inh")
plot(time,mean(res_s_n,0),color=sns.xkcd_rgb["greenish"],lw=3)

plot(time,zeros(len(time)),"k--",alpha=0.3)
fill_between([0,0.5],-1,1,color="gray",alpha=0.2)
fill_between([fixoff,fixoff+.35],-1,1,color="gray",alpha=0.2)

plot([time[time_silent]-W1/2,time[time_silent]-W1/2+W1],[-0.125,-0.125],color=sns.xkcd_rgb["orange"])
plot([time[time_react]-W1/2,time[time_react]-W1/2+W1],[-0.125,-0.125],color=sns.xkcd_rgb["greenish"])
plot([time[time_silent]-W1/2,time[time_silent]-W1/2],[-0.125,-0.12],color=sns.xkcd_rgb["orange"])
plot([time[time_silent]-W1/2+W1,time[time_silent]-W1/2+W1],[-0.125,-0.12],color=sns.xkcd_rgb["orange"])
plot([time[time_react]-W1/2,time[time_react]-W1/2],[-0.125,-0.12],color=sns.xkcd_rgb["greenish"])
plot([time[time_react]-W1/2+W1,time[time_react]-W1/2+W1],[-0.125,-0.12],color=sns.xkcd_rgb["greenish"])
print(time[time_silent]-W1/2,time[time_silent]-W1/2+W1)


ylabel("CCSI")
xlabel("time (s)")


xlim(-4.5,1)
xticks(arange(-4,2))
ylim(-0.2,0.2)
yticks(arange(-0.2,0.3,0.1))
tick_params(direction='in')
legend(frameon=False)

# interplay: firing rate x xcorrelation
subplot(2,2,3)
f = bz2.BZ2File('preprocessed_data/corr_xcorr_fr.pickle.bz2')
[res,pos_idx,neg_idx,zero]=load(f)
d= res[:,0,:]

FRS_pref=[]
PS_pref=[]

FRS_anti=[]
PS_anti=[]

zero=99
for i in pos_idx:
    # pref
    peak = mean(d[i][0][:,zero-1:zero+2],1)
    # each 100 bins has the same number: total spikes per delay
    fr = d[i][-2][:,0]
    FRS_pref.append(fr-mean(fr))
    PS_pref.append(peak-mean(peak))
    # anti
    peak = mean(d[i][1][:,zero-1:zero+2],1)
    # each 100 bins has the same number: total spikes per delay
    fr = d[i][-1][:,0]
    FRS_anti.append(fr-mean(fr))
    PS_anti.append(peak-mean(peak))


FRS_pref = concatenate(FRS_pref)
PS_pref = concatenate(PS_pref)
FRS_anti = concatenate(FRS_anti)
PS_anti = concatenate(PS_anti)


## compute errorbar from bootstrapping
R_pref = []
R_anti = []

for idx in bootstrap.bootstrap_indexes(FRS_pref,n_samples=n_perms):
    R_pref.append(spearmanr(FRS_pref[idx],PS_pref[idx])[0])

for idx in bootstrap.bootstrap_indexes(FRS_anti,n_samples=n_perms):
    R_anti.append(spearmanr(FRS_anti[idx],PS_anti[idx])[0])


R_pref = array(R_pref)
R_anti = array(R_anti)

r_pref = mean(R_pref)
r_anti = mean(R_anti)


pref_stdr = percentile(R_pref,[2.5,97.5])
anti_stdr = percentile(R_anti,[2.5,97.5])


pref_p = mean(R_pref<0)
anti_p = mean(R_anti>0)

## perm test
n_pref = len(FRS_pref)
FRS_both = concatenate([FRS_pref,FRS_anti])
PS_both = concatenate([PS_pref,PS_anti])
diffs=[]
for _ in range(n_perms):
    shuffle(FRS_both)
    r_p=spearmanr(FRS_both[:n_pref],PS_both[:n_pref])[0]
    r_a=spearmanr(FRS_both[n_pref:],PS_both[n_pref:])[0]
    diffs.append(r_p - r_a)
diff_p =  mean(array(diffs)>(r_pref - r_anti))

errorbar([0,1],[r_pref,r_anti],yerr=[pref_stdr-r_pref,anti_stdr - r_anti],color="black")

text(0.25,.25,"p=%.3f" % diff_p)
plot([0,1],[.25,.25],"k")
plot([0,1],[r_pref,r_anti],"k.",ms=20)
plot([-0.5,1.5],[0,0],"k--")
sns.despine()
ylabel("x-corr peak vs delay-fr")
xticks([0,1],["pref","anti-pref"])
ylim([-.2,0.3])
xlim([-0.25,1.25])
yticks([-.2,-.1,0,.1,.2,.3])
tight_layout()
tick_params(direction='in')


# FIRING RATE DIFFERENCE PREF VS ANTI-PREF FOR EXCITATORY AND INHIBITORY
subplot(2,2,4)

f = open("preprocessed_data/xcorr_fr_IE.pickle")
[time,diff_i_i,diff_fr_i,ci_i_i,ci_fr_i]= load(f)
f.close()

f = open("preprocessed_data/xcorr_fr_EE.pickle")
[time,diff_i_e,diff_fr_e,ci_i_e,ci_fr_e]= load(f)
f.close()


fill_between(time,ci_fr_e[:,0],ci_fr_e[:,1],alpha=0.3,color="gray",label="95% C.I.")
plot(time,diff_fr_e,color="k",lw=3,label="exc")
plot(time,diff_fr_i,"k--",lw=3,label="inh")
plot(time,zeros(len(time)),"k--",alpha=0.3)
fill_between([0,0.5],-15,15,color="gray",alpha=0.2)
fill_between([fixoff,fixoff+.35],-15,15,color="gray",alpha=0.2)


ylabel("firing rate pref - anti-pref \n (sp/s)")
xlim(-4.5,1)
ylim(-10,10)
legend(frameon=False)
yticks(arange(-10,11,5))
xticks(arange(-4,2))
xlabel("time (s)")
tick_params(direction='in')
sns.despine()


tight_layout(pad=1,h_pad=1, w_pad=1)

show()
