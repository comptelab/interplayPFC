'''
crosstemporal decoding matrix for Barbosa, Stein et al. (Nat Neuro, 2020)
created by H Stein, Mar 2020 

input: 	../data/meantf_crossmatrix_*.npy
		../data/crosstime.npy
output:	./crossmatrix_decoders.pkl
'''

from glob import glob
import numpy as np
import pickle
import sys
sys.path.insert(0,"../../helpers/")
import heike_helpers as hf
from joblib import Parallel, delayed
import multiprocessing

numcores = multiprocessing.cpu_count()


##############################################################################
#								PLOTTING	  				                 #
##############################################################################

matrix_ext		= 'meantf_crossmatrix_*'

matrix 			= sorted(glob('../../Data/decoders_and_behavior_EEG/'+matrix_ext))
print(matrix)
mtime 			= np.load('../../Data/decoders_and_behavior_EEG/'+'crosstime.npy')
'''
mtime encodes both alignment and time for each alignment piece of decoding matrix
mtime[1,:] is alignment with 
0: stimulus n-1; 1: probe n-1; 2: response n-1; 3: stimulus n
mtime[0,:] is time for each aligned piece
'''

##############################################################################
#									FUNCTIONS 				                 #
##############################################################################

def read_matrix(matrix):
	'''function reads trialwise tuning curves and calculates decoding strength 
	   as the cosine of the population vector'''
	# define angles that the tuning curves correspond to
	angs 	= np.linspace(0,2*np.pi-2*np.pi/8,8)

	# load data 
	print(matrix)
	matall 	= np.load(matrix)
	mat 	= np.zeros(matall.shape[2:4])

	# calculate mean decoding strength for each train/test time point
	for t in range(np.shape(matall)[-2]):
		for tt in range(np.shape(matall)[-1]):
			mat_vec		= np.array(list(map(lambda x: 
				  	  	  hf.circmean2(angs,x), matall[:,:,t,tt])))
			mat[t,tt] 	= np.mean(np.cos(mat_vec[:,0]))

	return mat


##############################################################################
#					READ DATA AND CALCULATE DECODING STRENGTH 	             #
##############################################################################

print('reading matrices')
matrix = np.array(Parallel(n_jobs=numcores)(delayed(read_matrix)(f) for f in matrix))


##############################################################################
#								SAVE FOR PLOTTING 	     			         #
##############################################################################

with open('../preprocessed_data/crossmatrix_decoders.pkl', 'wb') as f: pickle.dump([mtime, matrix], f, protocol=2)
