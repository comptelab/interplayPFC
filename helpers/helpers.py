
from __future__ import division
from numpy import *
from matplotlib.mlab import *
from circ_stats import *
from scipy.stats import *
from scikits.bootstrap import ci
from scikits import bootstrap
from scikits.bootstrap import bootstrap_indexes
import statsmodels.nonparametric.smoothers_lowess as loess
from constants import *
import seaborn as sns
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
import pandas as pd
import sys
from scipy import special
import heike_helpers as hf

find = lambda x: where(x)[0]


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


def cohen_d_ind(x,y):
        return (nanmean(x) - nanmean(y)) / sqrt((nanstd(x, ddof=1) ** 2 + nanstd(y, ddof=1) ** 2) / 2.0)

def cohen_d_rel(x,y):
        return (nanmean(x - y)) /nanstd(x-y, ddof=1)

def cohen_d(x,alpha=0):
        return (nanmean(x)-alpha) /nanstd(x, ddof=1)

def stderr(data):
	return std(data,0)/sqrt(len(data))
	
def boot_test(data,thr=0,n_samples=1000000):
	data=array(data)
	t_data = nanmean(data) - thr
	boot_data = data[bootstrap_indexes(data,n_samples=n_samples)]
	t_boot = (nanmean(boot_data,1) - nanmean(data)) 
	p =  nanmean(abs(t_data)<=abs(t_boot))
	return p,percentile(nanmean(boot_data,1),[2.5,97.5])

def boot_test1(data,thr=0,n_samples=1000000):
	data=array(data)
	t_data = nanmean(data) - thr
	boot_data = data[bootstrap_indexes(data,n_samples=n_samples)]
	t_boot = (nanmean(boot_data,1) - nanmean(data)) 
	low =  nanmean(t_data<=t_boot)
	high =  nanmean(t_data>=t_boot)
	return low,high,percentile(nanmean(boot_data,1),[2.5,97.5])


def perm_test2(data_a,data_b,n_perms=10000):
	n_a = len(data_a)
	r_d = mean(data_a) - mean(data_b)
	data_a = data_a.copy()
	data_b=data_b.copy()
	both = concatenate([data_a,data_b])
	D=[]
	for _ in range(n_perms):
		#idx=where(rand(len(both))<0.5)[0]
		shuffle(both)
		data_a,data_b=both[:n_a],both[n_a:]
		d=mean(data_a) - mean(data_b)
		D.append(d)
	return mean(r_d < array(D),0)*2


def remove_out(data):
	low,high = percentile(data,[2.5,97.5])
	low,high = percentile(data,[5,95])

	#return (data<=high) & (data>=low)
	return abs(data)<3*std(data)

def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	for s in sigs:
		beg =axis[s]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

def circ_mean(x):
	return circmean(x,low=-pi,high=pi)


def to_pi(angles):
	angles = array(angles)
	idx = angles>pi
	angles[idx] = angles[idx]-2*pi
	return angles



def compute_serial(report,target,d,xxx,flip=None):
	n=0
	err=circdist(report,target)
	m_err=[]
	std_err=[]
	count=[]
	cis=[]
	uf_err = err.copy()
	if flip:
		err = sign(d)*err
		d=abs(d)
	points_idx=[]
	for i,t in enumerate(xxx):
		# wi=w[i]
		idx=(d>=t)&(d<=t+w2)
		m_err.append(circ_mean(err[idx]))
		std_err.append(circstd(err[idx])/sqrt(sum(idx)))
		count.append(sum(idx))
		points_idx.append(idx)
	return [array(err),d,array(m_err),array(std_err),count,points_idx,n,uf_err]




def color_legend(colors,loc="best",ncol=1,fontsize=15):
	l=legend(frameon=False, loc=loc, ncol=ncol,fontsize=fontsize)
	for i,text in enumerate(l.get_texts()):
		text.set_color(colors[i])

	for item in l.legendHandles:
	    item.set_visible(False)
