'''
frequently used functions for Barbosa, Stein et al. (Nat Neuro, 2020)
created by H Stein, Mar 2020 
'''

import numpy as np
import scipy as sp
import scipy.stats as sps
import scipy.signal as spsi
import scipy.ndimage as spnd
import matplotlib.pyplot as plt
import math
import cmath
import pdb

#######################################################################

def len2(x):
	if type(x) is not type([]):
		if type(x) is not type(np.array([])):
			return -1
	return len(x)


#######################################################################

def phase2(x):
	if not np.isnan(x):
		return cmath.phase(x)
	return np.nan


#######################################################################

def circdist(angles1, angles2):
	if len2(angles2) < 0:
		if len2(angles1) > 0:
			angles2 = [angles2]*len(angles1)
		else:
			angles2 = [angles2]
			angles1 = [angles1]		
	if len2(angles1) < 0:
		angles1 = [angles1]*len(angles2)
	return np.array(list(map(lambda a1, a2: phase2(np.exp(1j*a1) / 
		np.exp(1j*a2)), angles1, angles2)))


#######################################################################

def circmean2(alpha,w):
    r = sum(w*np.exp(1j*alpha))
    return np.angle(r),np.abs(r)


#######################################################################

def filter_dat(dat, rt, iti, raderr, err):
    timeoutRT   = dat[dat.RT<rt]
    timeoutITI  = dat[dat.ITI<iti]
    raderror    = dat[dat.raderror<raderr]
    error       = dat[((dat.error<err) & (dat.error>-err))]
    return dat.iloc[timeoutRT.index & timeoutITI.index & raderror.index & error.index]


#######################################################################

def index_trials(beh, dec, info):
	'''match the trials from the beh pickle ("beh") and the trial info 
	used during decoding ("info"), and index each so that the trials 
	correspond between beh, decoder ("dec") and info
	'''
	beh.trial = beh.trial.astype('int')
	beh 	= beh.sort_values(by=['trial']).reset_index(drop=True)

	b_trial = beh.trial.values
	d_trial	= info[:,0].astype('int')

	# find the first trial of the second 1h session (trials in decoder 
	# are numbered 1:576 in each session), and adjust the index to 
	# beh index
	try:
		ind 	= np.where(d_trial-np.roll(d_trial,1) < 0)[0][1]
		d_trial = np.concatenate([d_trial[:ind],
				  d_trial[ind:]+np.ceil(d_trial[ind-1]/48.)*48])
	except: 
		d_trial = d_trial
	
	# find trials that pass exclusion criteria for both dec and beh
	b_ind 	= np.isin(b_trial, d_trial)
	d_ind 	= np.isin(d_trial, b_trial)

	beh 	= beh[b_ind].reset_index(drop=True)
	info	= info[d_ind,:]
	dec		= np.squeeze(dec[d_ind,:,:])

	# chech if trial information (presented stimulus angle) 
	# from matched beh and info corresponds
	ERR = np.sum(circdist(beh.target.values.astype('float'), 
		  np.deg2rad(info[:,1])))

	if ERR > .0005:
		print('misalignment: check stimuli', ERR)

	return beh, dec, ERR


#######################################################################

def serial_bias(target, serial, error, window, step, flip=None):
	xxx 	= np.arange(-np.pi, np.pi-window, step)
	d 		= circdist(serial, target)
	signd 	= np.repeat(1,len(d))
	c_err 	= np.zeros(np.shape(error))

	m_err=[]; std_err=[]; count=[]; points_idx = []

	if flip:
		xxx 	= np.arange(0, np.pi, step)
		signd 	= np.sign(d)
		error 	= signd * error
		d 		= np.abs(d)
	
	for t in xxx:
		idx = (d>=t) & (d<t+window) # sliding window 
		m_err.append(sps.circmean(error[idx],low=-np.pi,high=np.pi))
		std_err.append(sps.circstd(error[idx])/np.sqrt(np.sum(idx)))
		count.append(np.sum(idx))
		points_idx.append(idx)

		if sum(idx) > 0:
			c_err[idx] = signd[idx] * circdist(error[idx], 
				sps.circmean(error[idx],low=-np.pi,high=np.pi))

	return c_err, np.array(m_err), np.array(std_err), xxx, d, error

#######################################################################

def smooth_i(diff, subs, w1):   # i being a pair of neurons / one subject
	b = np.ones(w1)
	m = np.nanmean(diff[subs],0)
	p = spnd.filters.convolve1d(m, b/b.sum())
	return p


#######################################################################

def perm_test(a,b):
	d = np.nanmean(a-b)
	perms=[]
	for n in range(50000):
		idx = np.random.random(len(a))<0.5
		a[idx],b[idx] = b[idx],a[idx]
		d_perm= np.nanmean(a-b)
		perms.append(d_perm)
	return np.mean(d<=np.array(perms))