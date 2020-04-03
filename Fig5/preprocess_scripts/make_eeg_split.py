'''
behavior for Barbosa, Stein et al. (Nat Neuro, 2020)
created by H Stein, Mar 2020 

input: 	../data/meantf_dynamic_*.npy
		../data/dynamic_time.npy
		../data/behavior.pkl
output:	./split.pkl
'''
import sys
from glob import glob
import numpy as np
import scikits.bootstrap as boot
import pandas as pd
import pickle
sys.path.insert(0, '../../helpers/')
import heike_helpers as hf
from joblib import Parallel, delayed
import multiprocessing

numcores = multiprocessing.cpu_count()


##############################################################################
#					GET FILES, DECODER INFO AND BEHAVIOR	               	 #
##############################################################################

# files, decoder trial info and behavior
files 	= sorted(glob('../../Data/decoders_and_behavior_EEG/meantf_dynamic*'))
info 	= sorted(glob('../../Data/decoders_and_behavior_EEG/info_dynamic*'))
beh 	= pd.read_pickle('../../Data/decoders_and_behavior_EEG/behavior.pkl')
dtime 	= np.load('../../Data/decoders_and_behavior_EEG/dynamic_time.npy')
print(files)

# keep trials with RT<3 sec, prev ITI<5 sec, rad error<5 cm, ang error <1 rad
beh = hf.filter_dat(beh, rt=3, iti=5, raderr=5, err=1)

# remove 1st trials of each block 
beh = beh[beh.trial%48!=1]


##############################################################################
#									FUNCTIONS 				                 #
##############################################################################

def get_split(files):
	global beh
	print(files)

	# load data for each subject, only 1 and 3 sec trials
	datall 	= beh[(beh.delay!='0') & (beh.subject==files[0][-6:-4])]
	datall 	= datall.loc[:,datall.columns!='subject'].astype('float')
	decall	= np.load(files[0])
	infoall = np.load(files[1])

	# link behavior to decoder and check if alignment is correct (ERR!>0)
	dat, dec, ERR = hf.index_trials(datall, decall, infoall)
	print(np.unique(dat.delay), ERR>.005, dat.shape, dec.shape)

	# calculate trialwise decoding strength	
	s_bias 	= np.zeros([int(np.ceil(np.pi/(np.pi/20))),2,dtime.shape[0]])
	s_diff 	= np.zeros(dtime.shape[0])

	# calculate high- vs low-decoding serial bias for each time point
	for t in range(len(dtime)):
		# sliding average of tuning curves over 20 samples
		tuning 	= np.mean(dec[:,:,t-int(20/2):t+int(20/2)],2)

		# get trialwise population vector and its cosine (decoding strength)
		angs 		= np.linspace(0,2*np.pi-2*np.pi/8,8)
		tuning_vec 	= np.array(list(map(lambda x: hf.circmean2(angs,x), tuning)))
		strength 	= np.cos(tuning_vec[:,0])

		# define high- and low-decoding trials as above or below 75th percentile
		perc 	= np.percentile(strength,75)
		t_high 	= strength>perc
		t_low 	= strength<perc

		# calculate serial bias for high- and low-decoding trials separately
		_, high, _, _, dist_h, err_h = hf.serial_bias(dat.target[t_high].values, 
			dat.serial[t_high].values, dat.error[t_high].values, 
			window=np.pi/3, step=np.pi/20, flip=True)	
		_, low, _, _, dist_l, err_l = hf.serial_bias(dat.target[t_low].values, 
			dat.serial[t_low].values, dat.error[t_low].values, 
			window=np.pi/3, step=np.pi/20, flip=True)

		# save serial bias curves for high- and low-decoding trials separately
		s_bias[:,0,t] = high
		s_bias[:,1,t] = low

		# calculate difference between attractive biases (|theta d| <90ª) for this timepoint
		s_diff[t]	= np.mean(err_h[((dist_h>0)&(dist_h<np.pi/2))]) - \
					  np.mean(err_l[((dist_l>0)&(dist_l<np.pi/2))]) 	

	print(np.shape(s_bias), np.shape(s_diff))
	return s_bias, s_diff 


##############################################################################
#								RUN SPLITS 					                 #
##############################################################################

results = Parallel(n_jobs=numcores)(delayed(get_split)(f) for f in list(zip(files,info)))

sb_time = np.array([r[0] for r in results])
splits 	= np.array([r[1] for r in results])

tmax 		= np.where(dtime>-.84)[0][0]
tmin 		= np.where(dtime>.00)[0][0]


##############################################################################
#							SMOOTH DIST CURVE 				                 #
##############################################################################

idx 	= boot.bootstrap_indexes(splits,n_samples=10000)
ci_h 	= np.array(Parallel(n_jobs=numcores)(delayed(hf.smooth_i)(splits, i, 16) for i in idx))
split  	= np.mean(ci_h,0)
high	= np.array(list(map(lambda x: np.percentile(x, 97.5), ci_h.T)))
low		= np.array(list(map(lambda x: np.percentile(x, 2.5), ci_h.T)))


##############################################################################
#							SAVE FOR PLOTTING      						     #
##############################################################################

# with open('../preprocessed_data/split.pkl', 'wb') as f:
#     pickle.dump([sb_time, dtime, split, high, low, tmax, tmin], f, protocol=2)
