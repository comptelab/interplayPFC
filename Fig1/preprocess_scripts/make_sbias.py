from matplotlib.pylab import *
from joblib import Parallel, delayed  
import multiprocessing
import sys
sys.path.insert(0, '../../helpers/')
from circ_stats import *
from pickle import dump,load
import scikits.bootstrap as bootstrap
import difflib
import seaborn as sns
from scipy.stats import circmean
from joblib import Parallel, delayed  
import multiprocessing
warnings.filterwarnings("ignore")
import random

w1=pi/100
w2=pi/10
n_perms = 1000
xxx2=arange(0,pi-w2+w1,w1)


def circ_mean(x):
	return circmean(x,high=pi,low=-pi)


def compute_serial(report,target,d,xxx):
	err=circdist(report,target)
	idx_clean=(abs(err)<10*std((err)))
	err=err[idx_clean]
	d=d[idx_clean]
	m_err=[]
	std_err=[]
	count=[]
	cis=[]
	err = sign(d)*err
	d=abs(d)
	points_idx=[]
	for t in xxx:
		idx=(d>t)&(d<=t+w2)
		m_err.append(circ_mean(err[idx]))
		std_err.append(circstd(err[idx])/sqrt(sum(idx)))
		count.append(sum(idx))
		points_idx.append(idx)
	return array(err),d,array(m_err),array(std_err),count,points_idx

set_printoptions(precision=4)
sns.set_context("talk", font_scale=1.3)
sns.set_style("ticks")



f=open("../../Data/monkey_behavior_by_sessions.pickle","r")
session_close=load(f)
f.close()
sessions = unique([k for k in session_close.keys()])


# to account for eye tracking systematic errors:
# set cue location as the mean report location within a session 

cues_by_session ={}
mean_report_session = {}
for session in sessions:
	mean_report_session[session] = [[] for _ in range(len(sessions))]


for session in session_close.keys():
	trials=session_close[session]['trials']
	x = trials[:,3]
	y = trials[:,4]
	cues = trials[:,0]
	report = arctan2(y,x)
	for cue in range(1,9):
		idx = cues == cue
		mean_report_session[session][cue-1] += list(report[idx])
	mean_report_session[session] = amap(circmean,mean_report_session[session])


# compute trial error relative to mean report
# collect all info for serial biases calculation (e.g. previous stimulus distance, etc)

prev_curr = []
total_reports = []
total_cues=[]
for session in session_close.keys():
	i_close_trials = session_close[session]['i_close_trials']
	i_prev_trials = session_close[session]['i_prev_trials']

	prev_trials = session_close[session]['trials'][i_prev_trials,:]
	close_trials = session_close[session]['trials'][i_close_trials,:]

	prev_report=session_close[session]['report'][i_prev_trials]
	close_report=session_close[session]['report'][i_close_trials]
	
	prev_cue = (prev_trials[:,0]-1)/8.*(2*pi)
	curr_cue = (close_trials[:,0]-1)/8.*(2*pi)

	cues_by_session[session] = mean_report_session[session]

	curr_mean_reports = mean_report_session[session][array(close_trials[:,0],dtype='int')-1]
	prev_curr += list(circdist(prev_report,curr_mean_reports))
	total_reports+=list(close_report)
	total_cues+=list(curr_mean_reports)


prev_curr = array(prev_curr)
total_reports = array(total_reports)
total_cues = array(total_cues)



num_cores = multiprocessing.cpu_count()
boot_idx=bootstrap.bootstrap_indexes(total_reports,n_samples=n_perms)


def one_boot(i):
	err,d,m_err,std_err,count,points_idx=compute_serial(total_reports[i],total_cues[i],prev_curr[i],xxx2)
	return m_err

M = Parallel(n_jobs=num_cores)(delayed(one_boot)(i) for i in boot_idx)



err,d,m_err,std_err,count,points_idx=compute_serial(total_reports,total_cues,prev_curr,xxx2)

def one_perm(prev_curr):
	prev_curr = array(random.sample(prev_curr, len(prev_curr)))
	s=compute_serial(total_reports,total_cues,prev_curr,xxx2)
	return s[2]


perms = Parallel(n_jobs=num_cores)(delayed(one_perm)(prev_curr) for _ in xrange(n_perms))


stderr = std(M,0)
shuf_mean = mean(M,0)

down=(m_err-stderr)
up=(m_err+stderr)


f=open("../preprocessed_data/monkey_behavior.pickle","w")

dump([xxx2,down,up,w2,m_err,perms],f)
f.close()

