from glob import glob
import sys
import random as rd
import numpy as np
import scipy.io as scio
from joblib import Parallel, delayed
import multiprocessing as mp
import time as tme

##############################################################################
#                                     FILES                                  #
##############################################################################

counter         = int(sys.argv[1])
train_period    = sys.argv[2]
numcores        = mp.cpu_count()

stim            = sorted(glob('../../Data/alphas/*_alphaS.mat'))[counter]
probe           = sorted(glob('../../Data/alphas/*_alphaP.mat'))[counter]
resp            = sorted(glob('../../Data/alphas/*_alphaR.mat'))[counter]

print(stim, probe, resp, train_period)
if ((stim[:9] != resp[:9]) | (stim[:9] != probe[:9])): 
    print('files do not match'); kill  # make sure subjects match

nIter           = 100         # iterations of random sampling training set
window          = 5           # how many samples to average over


##############################################################################
#                                    FUNCTIONS                               #
##############################################################################

def find_discontinuity(trials):
    try:
        disc  = np.where(np.array(list(zip(trials[:-1], 
            trials[1:])))[:,1] - np.array(list(zip(trials[:-1], 
            trials[1:])))[:,0] < 0)[0][0] + 1
    except: disc=0
    return disc

def intersect_datasets(s,p,r):
    # intersect trials from the three files and select only common trials
    trials_s    = s['info'][:,0].astype('int')
    trials_p    = p['info'][:,0].astype('int')
    trials_r    = r['info'][:,0].astype('int')

    disc_s      = find_discontinuity(trials_s)
    disc_p      = find_discontinuity(trials_p)
    disc_r      = find_discontinuity(trials_r)

    trials_s[disc_s:] += 1000       # add number > maxtrials to have unique trial IDs
    trials_p[disc_p:] += 1000
    trials_r[disc_r:] += 1000

    _, i_s, i_p = np.intersect1d(trials_s, trials_p, return_indices=True)
    _,i_ps, i_r = np.intersect1d(trials_s[i_s], trials_r, return_indices=True)

    s['data']   = s['data'][i_s][i_ps];  s['info'] = s['info'][i_s][i_ps]
    p['data']   = p['data'][i_p][i_ps];  p['info'] = p['info'][i_p][i_ps]
    r['data']   = r['data'][i_r];        r['info'] = r['info'][i_r]

    if sum(sum(s['info'] != r['info']))>0: print('error concatenating'); kill 
    if sum(sum(s['info'] != p['info']))>0: print('error concatenating'); kill 
    # make sure concatenation gives correct result

    return s, p, r

def select_windows(s,p,r, i_s1, i_p, i_r, i_s2):
    s_time      = s['time'][0]
    p_time      = p['time'][0]
    r_time      = r['time'][0]

    t_s1        = np.where((s_time>i_s1[0])&(s_time<i_s1[1]))[0]
    t_p         = np.where((p_time>i_p[0])&(p_time<i_p[1]))[0]
    t_r         = np.where((r_time>i_r[0])&(r_time<i_r[1]))[0]
    t_s2        = np.where((s_time>i_s2[0])&(s_time<i_s2[1]))[0]

    data        = np.concatenate((s['data'][:-1,:,t_s1], p['data'][:-1,:,t_p],
                  r['data'][:-1,:,t_r], s['data'][1:,:,t_s2]), axis = 2)

    time        = np.stack((np.concatenate((s_time[t_s1], p_time[t_p], 
                  r_time[t_r], s_time[t_s2]), axis = 0), 
                  np.concatenate((np.zeros(len(t_s1)),np.zeros(len(t_p))+1,
                  np.zeros(len(t_r))+2,np.zeros(len(t_s2))+3),axis = 0)))
 
    info        = s['info']

    return data, time, info

def select_consecutive(data, info):
    valid       = np.where(info[:,0] == np.roll(info[:,0],-1)-1)[0]
    data        = data[valid]
    info        = info[valid]

    return data, info

def downsample_data(data, time, window):
    downsampled = np.zeros([data.shape[0],data.shape[1],int(data.shape[2]/window)])
    time        = time[:,window::window]

    for s in range(int(data.shape[2]/window)):
        downsampled[:,:,s] = np.mean(data[:,:,s*window:(s+1)*window],2)

    return downsampled, time

def select_train_data(bins):
    _, bincnt   = np.unique(bins, return_counts = True)
    nbin        = bincnt.min()
    trainid     = []

    for b in range(bins.max()):
        trainid.append(rd.sample(list(np.where(bins == b+1)[0]), nbin))

    return np.array(trainid)

def cross_IEM(U_train, M_train, U_test, bin_test):
    if train_period == 'delay':
        ttrain      = np.where((time[1,:] == 0) & 
                          (time[0,:] > .5) & (time[0,:] < 1))[0]
    elif train_period == 'response':
        ttrain      = np.where((time[1,:] == 2) & 
                          (time[0,:] > -.25) & (time[0,:] < .25))[0]
    elif train_period == 'fixation':
        ttrain      = np.where((time[1,:] == 3) & 
                          (time[0,:] > -1.1) & (time[0,:] < -.6))[0]

    U1      = np.mean(U_train[:,:,ttrain],2).T
    M1      = M_train.T
    W       = np.dot(np.dot(U1,M1.T),np.linalg.inv(np.dot(M1,M1.T)))
    M_pred  = np.zeros([M1.shape[0], U_test.shape[1]])

    Mval  = np.dot(np.linalg.inv(np.dot(W.T,W)),np.dot(W.T,np.mean(U_test[:,ttrain],1)))
    Mval  = np.roll(Mval, -(bin_test-1))

    for tt in range(U_test.shape[1]):
        U2  = U_test[:,tt]
        M2  = np.dot(np.linalg.inv(np.dot(W.T,W)),np.dot(W.T,U2))
        M2  = np.roll(M2, -(bin_test-1))
        M_pred[:,tt] = M2

    return W, M_pred

def run_trialwise(i):
    print(i)
    tf  = np.zeros([nTrials,nBins,nSamps])
    weights = np.zeros([nTrials,nElectrodes,nBins])

    for trial in range(nTrials):
        # split train and test set
        U_train     = data[np.arange(data.shape[0]) != trial]
        bin_train   = bins[np.arange(data.shape[0]) != trial]
        U_test      = data[trial]
        bin_test    = bins[trial]

        setid       = select_train_data(bin_train)

        # average over trials for training data
        traindata   = []
        for b in range(bins.max()):
            traindata.append(np.mean(U_train[setid[b]],0))

        U_train     = np.array(traindata)
        bin_train   = np.unique(bin_train)
        M_train     = np.zeros([nBins,nBins])

        for b in bin_train:
            M_train[b-1] = np.roll(basisFun,b-1)

        weights[trial,:,:], tf[trial,:,:] = \
            cross_IEM(U_train, M_train, U_test, bin_test)


    return weights, tf

##############################################################################
#                              MAIN SCRIPT                                   #
##############################################################################

eeg_s = scio.loadmat(stim)             # stim-aligned
eeg_p = scio.loadmat(probe)            # probe-aligned
eeg_r = scio.loadmat(resp)             # resp-aligned

# select only trials present in all datasets
eeg_s, eeg_p, eeg_r = intersect_datasets(eeg_s, eeg_p, eeg_r)

# select time windows of interest
data, time, info    = select_windows(eeg_s, eeg_p, eeg_r, 
                      [-.5,2], [-2,.5], [-.5,.5], [-1.5,.5])

# select only consecutive trials
data, info          = select_consecutive(data, info)
del eeg_s; del eeg_r; del eeg_p

# average over windows of 5 samples
data, time          = downsample_data(data, time, window)

# dimensions
nTrials         = data.shape[0]
nElectrodes     = data.shape[1]
nSamps          = data.shape[2]

# bins and basis function
bins            = info[:,2] # select column containing first trials stimulus
nBins           = np.max(bins)
x               = np.linspace(0, 2*np.pi-2*np.pi/nBins, nBins)
basisFun        = np.abs(np.sin(x/2 + np.pi/2)**7)

# run nIter iterations of IEM
ti = tme.time()
results = Parallel(n_jobs=numcores)(delayed(run_trialwise)(i) for i in range(nIter))
print(tme.time() - ti)

weights = np.array([r[0] for r in results]) 
tf      = np.array([r[1] for r in results])     

tf_mean = np.mean(tf,0)
print(np.shape(tf_mean))

# save stuff
name = train_period + '__iter-%.0f_' % nIter

np.save('../../Data/decoders_and_behavior_EEG/meantf_' + name + stim[-14:-11], tf_mean)
np.save('../../Data/decoders_and_behavior_EEG/info_' + name + stim[-14:-11], info)
#    np.save('../../Data/decoders_and_behavior_EEG/weights_' + name + stim[-14:-11], weights)
#    np.save('../../Data/decoders_and_behavior_EEG/time', time)
