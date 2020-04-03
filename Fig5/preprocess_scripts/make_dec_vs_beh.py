
from matplotlib.pylab import *
from numpy import *
from pickle import load
from scikits import bootstrap
import sys
from scipy.stats import *
from joblib import Parallel, delayed
import multiprocessing
from scipy.signal import gaussian
from scipy.ndimage import filters
from pickle import dump
import bz2

num_cores = multiprocessing.cpu_count()

w1=pi/20
w2=pi/3

w_smooth = 3

flip=True
xxx=arange(0,pi-w2+w1,w1)

f = bz2.BZ2File("../preprocessed_data/final_0.05_1.0leave_one_out.pickle.bz2")
[time,z]=load(f)

time-=1

f=open("../preprocessed_data/beh.pickle")
prev_curr,beh_pairs=load(f)

npairs= len(beh_pairs)

#best X% computed with leave one out
n = map(mean,mean(array(z)[(time<-3)],0))
good_pairs=n<percentile(n,33)


# use all pairs
#good_pairs = range(npairs)

cuts = [25] 
cut = 25


OTHERS = True


def circ_mean(x):
        return circmean(x,low=-pi,high=pi)

def perm_test(a,b):
	d = nanmean(a)-nanmean(b)
	c = concatenate([a,b])
	n_tot = len(c)
	idx = range(n_tot)
	perms=[]
	for n in range(1000):
		shuffle(idx)
		d_perm = nanmean(c[idx[:n_tot/2]]) - nanmean(c[idx[n_tot/2:]])
		perms.append(d_perm)
	return mean(d<=perms)

def compute_serial(err,d):
	idx=abs(err)<radians(45)
	err=err[idx]
	d=d[idx]
	m_err=[]
	std_err=[]
	count=[]
	cis=[]
	if flip:
		err = sign(d)*err
		d=abs(d)
	points_idx=[]
	for t in xxx:
		idx=(d>=t)&(d<=t+w2)
		m_err.append(circ_mean(err[idx]))
		std_err.append(std(err[idx])/sqrt(sum(idx)))
		count.append(sum(idx))
		points_idx.append(idx)
	return array(err),d,array(m_err),array(std_err),count,points_idx


def one_time(zi):
	print zi
	difs=zeros((npairs,2))
	serials=zeros((npairs,4,len(xxx)))
	trials=zeros((npairs,2))
	for n,p in enumerate(z[zi]):

		# behavior report and prev-curr when recording ensemble
		b = beh_pairs[n]
		ps = prev_curr[n]

		# low decoding error for these trials
		low=percentile(p,25)
		
		# low-error trials bias
		idx=(p<=low) 
		difs[n,0]=(nanmean(b[idx][ps[idx]<0])-nanmean(b[idx][ps[idx]>0]))
		err1,d,m_err1,std_err,count,points_idx=compute_serial(b[idx],ps[idx])

		# all other trials
		idx=p>=low
		difs[n,1]=(nanmean(b[idx][ps[idx]<0])-nanmean(b[idx][ps[idx]>0]))
		err2,d,m_err2,std_err,count,points_idx=compute_serial(b[idx],ps[idx])

		serials[n,0]=m_err1
		serials[n,1]=m_err2

		
	return difs,serials,trials


difs = Parallel(n_jobs=num_cores)(delayed(one_time)(zi) for  zi in range(sum([time<1])))
time=time[time<1]
serial = array([d[1] for d in difs])
difs = array([d[0] for d in difs])



time+=0.5
xxx = degrees(xxx)
w2=degrees(w2)

serial = degrees(serial)
difs = degrees(difs)

# compute p values of permutation test
a=difs[:,good_pairs,0]
b=difs[:,good_pairs,1]
ps =  Parallel(n_jobs=num_cores)(delayed(perm_test)(b[t],a[t]) for t in range(len(time)))


# smoth split through time
def smooth_i(h,i):
	m=nanmean(h[:,i],1)
	p=filters.convolve1d(m, b/b.sum())
	return p

#negative diff -> attraction
h=-1*(difs[:,good_pairs,0]-difs[:,good_pairs,1]) #

b = gaussian(5,5)
idx=bootstrap.bootstrap_indexes(h[0],n_samples=10000)
res = Parallel(n_jobs=num_cores)(delayed(smooth_i)(h,i) for i in idx)
ci_h = array(res)


f=open("../preprocessed_data/0.05_1.0_beh_vs_dec_others.pickle","w")
dump([time,ci_h,res,ps,serial,good_pairs,xxx,w2],f)
f.close()
