'''
behavior for Barbosa, Stein et al. (Nat Neuro, 2020)
created by H Stein, Mar 2020 

input: 	../data/behavior.pkl
output:	./serial_bias.pkl
'''

import numpy as np
import scikits.bootstrap as boot
import pandas as pd
import pickle
import sys
sys.path.insert(0, '../../helpers/')
import heike_helpers as hf


##############################################################################
#                                  READ DATA                                 #
##############################################################################

alldat = pd.read_pickle('../../Data/decoders_and_behavior_EEG/behavior.pkl')
print(alldat.groupby('subject').count())


##############################################################################
#                   REMOVE OUTLIERS AND CALCULATE SERIAL BIAS                #
##############################################################################

# keep trials with RT<3 sec, prev ITI<5 sec, rad error<5 cm, ang error <1 rad
dat = hf.filter_dat(alldat, rt=3, iti=5, raderr=5, err=1)

# remove 1st trials of each block 
dat = dat[dat.trial%48!=1]

# calculate serial bias for each subject, only for trials with 1 or 3 sec delay
serial = []
for subj in dat.subject.unique():
    ddat = dat[dat.subject == subj]
    ddat = ddat[ddat.delay != '0']
    _, sb, _, xxx, _, _ = hf.serial_bias(ddat.target.values, ddat.serial.values, 
        ddat.error.values, window=np.pi/3, step=np.pi/100, flip=True)
    serial.append(sb)

# transform to degrees
serial = np.rad2deg(serial)
xxx = np.rad2deg(xxx)

# CI across subjects and points with significantly attractive or repulsive bias
ci     = np.array(list(map(lambda x: boot.ci(x), serial.T)))
sigpos = ci[:,0]>0
signeg = ci[:,1]<0


##############################################################################
#                          SAVE FOR PLOTTING                                 #
##############################################################################

with open('../preprocessed_data/serial_bias.pkl', 'wb') as f:
    pickle.dump([xxx, serial, sigpos, signeg], f, protocol=2)
