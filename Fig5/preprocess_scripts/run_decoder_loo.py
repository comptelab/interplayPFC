from __future__ import division
from joblib import Parallel, delayed  
import sys
sys.path.insert(0, '../../helpers/')
from circ_stats import *
import statsmodels.api as sm
from multiprocessing import Value
from math import atan2
from sklearn.model_selection import train_test_split
from pickle import load,dump
import multiprocessing
warnings.filterwarnings("ignore")
num_cores = multiprocessing.cpu_count()


f=open("../../Data/monkey_behavior_by_sessions.pickle","r")
session_close=load(f)
f.close()

f=open("../../Data/neural_behavior_by_neuron.pickle","r")
data=load(f)
f.close()

f=open("../../Data/neural_data_concat.pickle","r")
data_c=load(f)
f.close()


n_perms = 1000
nneurons = len(data)
kfolds = 50

w1=0.05
w2=1

time=arange(-5,5,w1)
n_time = len(time)



def fit_pred_one(spikes_train, spikes_test, beh_train, beh_test):
	X = column_stack([ones(shape(spikes_train)[0]),spikes_train])
	Y = column_stack([cos(beh_train),sin(beh_train)])
	model = sm.OLS(Y, X)
	fit=model.fit()
	X = concatenate([[1],spikes_test])
	p = fit.predict(X)
	x = p[:,0]
	y = p[:,1]
	pred=atan2(y,x)
	acc=abs(circdist(pred,beh_test)[0])
	return acc

def fit_pred_all2(p,t):
	beh = prevstim_ang[p]
	spikes = counts_ens[p][:,t,:]
	preds=[]
	idx=arange(len(beh))
	perms = []
	for i in idx:
		idx_m = delete(idx,i)
		fits=[]
		spikes_train, spikes_test, beh_train, beh_test = spikes[idx_m], spikes[i], beh[idx_m], beh[i]
		pred=fit_pred_one(spikes_train, spikes_test, beh_train, beh_test)
		preds.append(pred)
		perm=[]
		for _ in range(n_perms):
			shuffle(beh_train)
			pred=fit_pred_one(spikes_train, spikes_test, beh_train, beh_test)
			perm.append(pred)

		perms.append(perm)
	if p == 0: print("%i%%" % (t/n_time*100))
	
	# compute decoder distance from shuffle (z-score) in stdevs
	return (preds - mean(perms))/std(perms)


ensembles={}
prevstim = [[]]*nneurons
counts = [[[] for _ in range(n_time)] for _ in range(nneurons)]

#  (by neurons) spike counts in sliding windows of w2, steps of w1 and stimuli to decode
for n in range(nneurons):
	session=data[n]["session"]

	# Select ensembles from session name to decode from
	if session in ensembles.keys():
		ensembles[session].add(n)
	else:
		ensembles[session]=set([n])
		
	# index of consequetive trials
	i_close_trials = session_close[session]["i_close_trials"]

	# index of consequetive trials (aligned to previous)
	i_prev_trials = session_close[session]["i_prev_trials"]

	# spike times (aligned to current trial)
	spikes=data_c[n]
	
	# previous-trial stimuli
	indx_p=data[n]["INDX"][i_prev_trials,:]
	prevstim[n] = list(indx_p[:,0])

	for i,beg in enumerate(time):
		end = min(beg + w2,4.5)
		spike_count = nansum((spikes > beg) & (spikes <= end),1)
		counts[n][i] = spike_count

# reorganize spike counts and stimuli to decode by ensembles (ensembles)
counts_ens = []
prevstim_ens = []
for i in range(len(ensembles.values())):
	ps = list(ensembles.values()[i])
	curr_trials = array([counts[n] for n in ps]).T
	counts_ens.append(curr_trials)
	prevstim_ens += [prevstim[ps[0]]]

# convert stim (1,2,..) to radians
prevstim_ang = amap(lambda cue: (array(cue)-1)/8.0*(2*pi), prevstim_ens)

print "decoding from #%.2f networks" %(len(ensembles))

res= [Parallel(n_jobs=num_cores)(delayed(fit_pred_all2)(p,t) for p in range(len(ensembles))) for t in range(n_time)]

f=open("../preprocessed_data/loo_dec.pickle","w")
dump([time,res],f)

