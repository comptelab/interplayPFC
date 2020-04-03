
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
find = lambda x: where(x)[0]
import heike_helpers as hf


set_printoptions(precision=4)
sns.set_context("talk", font_scale=1.3)
sns.set_style("ticks")
sns.set_style({"ytick.direction": "in"})

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

# def perm_test(data_a,data_b,n_perms=10000):
# 	r_d = mean(data_a - data_b,0)
# 	data_a = data_a.copy()
# 	data_b=data_b.copy()
# 	D=[]
# 	for _ in range(n_perms):
# 		idx=where(rand(len(data_a))<0.5)[0]
# 		data_a[idx],data_b[idx]=data_b[idx],data_a[idx]
# 		d=mean(data_a - data_b,0)
# 		D.append(d)
# 	return mean(r_d < array(D),0)*2

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


def rem_sys_err2(report,target,frac=.25):
	# n_trials = len(report)
	# report = repeat(report,3)
	# target = repeat(target,3)

	fit_resp = loess.lowess(report, target, frac = frac, return_sorted = False)
	fit_error	= circdist(fit_resp, target)
	clean_res = circdist(report, fit_error)
	return clean_res,fit_resp

	return clean_res[n_trials:2*n_trials],fit_resp[n_trials:2*n_trials]


def rem_sys_err3(report,target,frac=.25):

	errors = circdist(target,report)
	stimulus_angles = target

	n_trials = len(stimulus_angles)
	x = array([stimulus_angles,
	                                 stimulus_angles + radians(360),
	                                 stimulus_angles + radians(720)]).flatten()
	y = tile(errors, 3)
	smoothed = loess.lowess(x, y, frac = frac, return_sorted = False) 

	sys_error = squeeze(array(smoothed)[n_trials:n_trials * 2])

	return sys_error

def rem_sys_err4(report,target,frac=.25):
	n_trials = len(report)
	report2 = tile(report,3)
	target2 = array([target, target+2*pi, target+4*pi]).flatten()
	fit_resp = loess.lowess(report2, target2, frac = frac, return_sorted = False)
	fit_error	= circdist(fit_resp, target2)
	clean_res = circdist(report2, fit_error)

	return clean_res[n_trials:n_trials * 2],fit_resp[n_trials:n_trials * 2]


def compute_seria_from_pandas(pandas,xxx,flip=None):
	return compute_serial(pandas.report_angle.values,pandas.target_angle.values,
		pandas.prev_curr,xxx,flip)

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

def one_subject(report,target,prev_curr):

	total_t = len(report)

	#sbias_all = compute_serial(report,target,prev_curr,xxx,flip)
	sbias_all = hf.folded_bias(prev_curr, circdist(report, target), w2, w1)

	if do_thirds:
		#sbias_first = compute_serial(report[:int(total_t/3)],target[:int(total_t/3)],prev_curr[:int(total_t/3)],xxx,flip)
		sbias_first = hf.folded_bias(prev_curr[:int(total_t/3)], circdist(report[:int(total_t/3)], target[:int(total_t/3)]), w2, w1)
		#sbias_last = compute_serial(report[2*int(total_t/3):],target[2*int(total_t/3):],prev_curr[2*int(total_t/3):],xxx,flip)
		sbias_last = hf.folded_bias(prev_curr[2*int(total_t/3):], circdist(report[2*int(total_t/3):], target[2*int(total_t/3):]), w2, w1)
	else:
		#sbias_first = compute_serial(report[:int(total_t/2)],target[:int(total_t/2)],prev_curr[:int(total_t/2)],xxx,flip)
		sbias_first = hf.folded_bias(prev_curr[:int(total_t/2)], circdist(report[:int(total_t/2)], target[:int(total_t/2)]), w2, w1)
		#sbias_last = compute_serial(report[int(total_t/2):],target[int(total_t/2):],prev_curr[int(total_t/2):],xxx,flip)
		sbias_last = hf.folded_bias(prev_curr[int(total_t/2):], circdist(report[int(total_t/2):], target[int(total_t/2):]), w2, w1)

	#sbias_med = compute_serial(report[int(total_t/3):2*int(total_t/3)],target[int(total_t/3):2*int(total_t/3)],prev_curr[int(total_t/3):2*int(total_t/3)],xxx,flip)
	sbias_med = hf.folded_bias(prev_curr[int(total_t/3):2*int(total_t/3)], circdist(report[int(total_t/3):2*int(total_t/3)], target[int(total_t/3):2*int(total_t/3)]), w2, w1)


	return sbias_all,sbias_first,sbias_last,sbias_med


def one_subject_future(report,target,prev_curr,err,idx,n_shifts=100):
	m_f_s=[]
	flip_err_s=[]
	errs_s=[]
	m_f_s_first = []
	m_f_s_last = []


	for f in range(1,n_shifts):
		prev_curr_f = circdist(roll(target,-f),target)
		sbias,sbias_first,sbias_last,errs,prevs,_ = one_subject(report,target,prev_curr_f,err,idx,trial_x,False)

		m_f_s.append(sbias[2])
		m_f_s_first.append(sbias_first[2])
		m_f_s_last.append(sbias_last[2])
		flip_err_s.append(sbias[0])
		errs_s.append(errs)

	return mean(m_f_s,0),mean(m_f_s_first,0),mean(m_f_s_last,0),mean(flip_err_s,0), mean(errs_s,0)

def plot_sigs(all_s_a,color,all_s_b=[],upper=[3.75,4],alpha=0.05,pvalues=[]):
	if len(pvalues)==0:
		n_perms = 10000
		n_subjects_a = len(all_s_a)
		if len(all_s_b):
			ci_sb = array([ci(sb,method="pi",n_samples=10000,alpha=alpha) for sb in (all_s_a-all_s_b).T])
		else:
			ci_sb = array([ci(sb,method="pi",n_samples=10000,alpha=alpha) for sb in (all_s_a).T])
		sigs = find((ci_sb[:,0]>0) | ((ci_sb[:,1])<0))
	else:
		sigs = find(array(pvalues)<alpha)
	ylim(-2,upper[1])
	sig_bar(sigs,xxx2,upper,color)

def plot_serial(all_s,color,label=None,xk=None,nan=False):
	mean = np.mean
	if nan:
		mean = nanmean
	if xk is None:
		xx = xxx2
	else:
		xx = xk
	stderr = array([ci(sb,statfunction=mean,alpha=1-0.68,method="pi") for sb in (all_s).T])
	if not label:
		fill_between(xx,degrees(stderr[:,0]),degrees(stderr[:,1]),color=color,alpha=0.2)
	else:
		fill_between(xx,degrees(stderr[:,0]),degrees(stderr[:,1]),color=color,alpha=0.2,label=label)
	plot(xx,degrees(mean(all_s,0)),color=color)
	plot(xx,zeros(len(xx)),"k--",alpha=0.5)
	if type_ori:
		xlabel(r"relative orientation of previous trial ($^\circ$)")
	else:
		xlabel(r"relative color of previous trial ($^\circ$)")
	ylabel(r"error on current trial ($^\circ$)")
	#legend()
	sns.despine()
	ylim(-2,3)
	#xlim((xx[0],xx[-1]))
	#xticks([xx[0],xx[-1]/2,xx[-1]])

def plot_one_exp(all_M,all_M_first,all_M_last,flip_err,uf_err,SBIAS_past,ERRS,save_fig=False):
	
	all_M=array(all_M)

	all_M_first=array(all_M_first)
	all_M_last=array(all_M_last)

	flip_err=array(flip_err)

	figure(figsize=(15,5))

	# serial biases for all trials
	subplot(1,3,1)
	plot_serial(all_M,"k")
	plot_sigs(all_M,"k")


	# plot_serial(all_M,"k")
	# plot_sigs(all_M,"k")


	#serial biases for early and late
	subplot(1,3,2)
	plot_serial(all_M_first,"k","first third")
	plot(xxx2,degrees(mean(all_M,0)),"k--")
	plot_serial(all_M_last,"g","last third")
	plot_sigs(all_M_first,"g",all_M_last)


	subplot(1,3,3)
	cis=degrees([ci(s,output="errorbar",method="pi") for s in array(SBIAS_past).T])
	means=degrees(mean(SBIAS_past,0))

	bar(past_times,means,color="black")

	n_subjects = shape(SBIAS_past)[0]
	for p_i,p in enumerate(past_times):
		errorbar(p,means[p_i],cis[p_i],color="r")
		# sbias = array(SBIAS_past)[:,p_i]
		# plot(p*ones(n_subjects),degrees(sbias),"r.",ms=15,alpha=0.5)

	errorbar(p,means[p_i],cis[p_i],color="r",label="95% C.I.")
	plot(past_times,zeros(len(past_times)),"k--")

	ylabel(r"absolute serial biases ($^\circ$)")
	xlabel("n-back")
	xlim((0,11))
	ylim(-0.5,1.5)
	xticks(range(1,11))

	sns.despine()

	if save_fig:
		savefig(sys.argv[0]+"1.svg",dpi=300)


	regres = amap(lambda x: linregress(range(len(x)),x),flip_err)


	uf_regres = amap(lambda x: linregress(range(len(x)),x),abs(array(uf_err)))

	return regres, uf_regres


def color_legend(colors,loc="best",ncol=1,fontsize=15):
	l=legend(frameon=False, loc=loc, ncol=ncol,fontsize=fontsize)
	for i,text in enumerate(l.get_texts()):
		text.set_color(colors[i])

	for item in l.legendHandles:
	    item.set_visible(False)
