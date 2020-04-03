'''
crosstemporal decoders for Barbosa, Stein et al. (Nat Neuro, 2020)
created by H Stein, Mar 2020 

input: 	../data/meantf_delay_*.npy
		../data/meantf_response_*.npy
		../data/time.npy

output:	./cross_decoders.pkl
'''

from glob import glob
import numpy as np
import scikits.bootstrap as boot
import pickle
import sys
sys.path.insert(0,'../../helpers/')
import heike_helpers as hf
from joblib import Parallel, delayed
import multiprocessing

numcores = multiprocessing.cpu_count()


##############################################################################
#									PATHS/FILES	      		             	 #
##############################################################################

# files for delay and response decoder, and time
delay_ext		= 'meantf_delay_*'
resp_ext		= 'meantf_response_*'

delay		  	= np.array(sorted(glob('../../Data/decoders_and_behavior_EEG/'+delay_ext), key=lambda f: f[-7:]))
resp		  	= np.array(sorted(glob('../../Data/decoders_and_behavior_EEG/'+resp_ext), key=lambda f: f[-7:]))
dtime 			= np.load('../../Data/decoders_and_behavior_EEG/'+'time.npy')
print(delay, resp)


##############################################################################
#					FUNCTIONS FOR DECODING STRENGTH & TUNING	             #
##############################################################################

def read_decoders(dec):
	'''function reads trialwise tuning curves and calculates decoding strength 
	   as the cosine of the population vector'''
	# define angles that the tuning curves correspond to
	angs 		= np.linspace(0,2*np.pi-2*np.pi/8,8)

	# load data and calculate mean decoding strength
	decall 		= np.load(dec)
	print(dec)

	# iterate through trials and get cosine 
	cos=[]
	for t in range(np.shape(decall)[0]):
		dec_vec = np.array(list(map(lambda x: hf.circmean2(angs,x), 
				  decall[t,:,:].T)))
		cos.append(np.cos(dec_vec[:,0]))

	return np.mean(cos,0)


def read_tunings(dec):
	'''function loads and averages trialwise tuning curves'''
	print(dec)
	tuning 			= np.mean(np.load(dec),0)

	return tuning


##############################################################################
#								READ DATA 					                 #
##############################################################################

print('reading decoders')
d_delay = np.array(Parallel(n_jobs=numcores)(delayed(read_decoders)(f) for f in delay))
d_resp = np.array(Parallel(n_jobs=numcores)(delayed(read_decoders)(f) for f in resp))

print('reading tuning curves')
t_delay = np.array(Parallel(n_jobs=numcores)(delayed(read_tunings)(f) for f in delay))
t_resp = np.array(Parallel(n_jobs=numcores)(delayed(read_tunings)(f) for f in resp))


##############################################################################
#					CI AND SIGNIFICANT DELAY DECODING	      		       	 #
##############################################################################

# CI for significant delay code
ci99_5  = np.array(list(map(lambda x: boot.bootstrap.ci(x, alpha=.005), d_delay.T)))
sig_005 = ci99_5[:,0]>0


##############################################################################
# 								SAVE FOR PLOTTING 							 #
##############################################################################

with open('../preprocessed_data/cross_decoders.pkl', 'wb') as f:
     pickle.dump([dtime, d_delay, d_resp, sig_005, t_delay], f, protocol=2)
