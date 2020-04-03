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
numcores        = 3#mp.cpu_count()

stim         = sorted(glob('../../Data/alphas/*_alphaS.mat'))[counter]
print(stim)

nIter           = 100          # iterations of random sampling training set
window          = 5            # how many samples to average over


##############################################################################
#                                    FUNCTIONS                               #
##############################################################################

def select_windows(s, i_s):
    s_time      = s['time'][0]
    t_s         = np.where((s_time>i_s[0])&(s_time<i_s[1]))[0]
    data        = s['data'][:,:,t_s]
    time        = s_time[t_s]
    info        = s['info']

    return data, time, info

def downsample_data(data, time, window):
    downsampled = np.zeros([data.shape[0],data.shape[1],int(data.shape[2]/window)])
    time        = time[window::window]

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

def diagonal_IEM(U_train, M_train, U_test, bin_test):
    W_est   = np.zeros([U_test.shape[1], M_train.shape[0], U_test.shape[0]])
    M_pred  = np.zeros([M_train.shape[0], U_test.shape[1]])
    
    for t in range(U_test.shape[1]):
        U1  = U_train[:,:,t].T
        M1  = M_train.T
        W   = np.dot(np.dot(U1,M1.T),np.linalg.inv(np.dot(M1,M1.T)))
        W_est[t,:,:] = W.T

        U2  = U_test[:,t]
        M2  = np.dot(np.linalg.inv(np.dot(W.T,W)),np.dot(W.T,U2))
        M2  = np.roll(M2, -(bin_test-1))
        M_pred[:,t] = M2

    return W_est, M_pred

def run_trialwise(i,nTrials):
    print(i)
    tf      = np.zeros([nTrials,nBins,nSamps])
    weights = np.zeros([nTrials,nElectrodes])

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

        w, tf[trial,:,:] = diagonal_IEM(U_train, 
                M_train, U_test, bin_test)

    return weights, tf

##############################################################################
#                              MAIN SCRIPT                                   #
##############################################################################

eeg_s = scio.loadmat(stim)             # stim-aligned

# select time windows of interest
data, time, info = select_windows(eeg_s, [-1.5,.5])
del eeg_s

# average over windows of 5 samples
data, time  = downsample_data(data, time, window)
dtidx     	= np.mean(np.var(data,1),1) < (np.mean(np.mean(np.var(data,1),1)) \
              + 4*np.std(np.mean(np.var(data,1),1)))
data 		= data[dtidx]
info 		= info[dtidx]

# dimensions
nTrials         = data.shape[0]
nElectrodes     = data.shape[1]
nSamps          = data.shape[2]

# bins and basis function
bins            = info[:,4] # select column containing previous stimulus
nBins           = np.max(bins)
x               = np.linspace(0, 2*np.pi-2*np.pi/nBins, nBins)
basisFun        = np.abs(np.sin(x/2 + np.pi/2)**7)

# run nIter iterations of IEM
ti = tme.time()
results = Parallel(n_jobs=numcores)(delayed(run_trialwise)(i,nTrials) for i in range(nIter))
print(tme.time() - ti)

weights = np.array([r[0] for r in results]) 
tf 		= np.array([r[1] for r in results])     

tf_mean = np.mean(tf,0)
print(np.shape(tf_mean))

# save stuff
name    = 'dynamic_' + '_iter-%.0f_' % nIter
np.save('../../Data/decoders_and_behavior_EEG/meantf_' + name + stim[-14:-11], tf_mean)
np.save('../../Data/decoders_and_behavior_EEG/info_' + name + stim[-14:-11], info)
# np.save('../../Data/decoders_and_behavior_EEG/weights_' + name + stim[-14:-11], weights)
# np.save('../../Data/decoders_and_behavior_EEG/dynamic_time', time)
