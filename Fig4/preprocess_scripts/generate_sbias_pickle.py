from __future__ import division
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../helpers/')
from circ_stats import *
import glob
from pickle import dump


thr = 20 #45
files = glob.glob("../preprocessed_data/*txt")
files.sort()
M=[]
S=[]
E=[]
for si,fname in enumerate(files):

	bkg = fname.split(".txt")[0].split("read_out_log_")[1]
	f=open(fname)
	s=f.readlines()

	d=[]
	for i,s1 in enumerate(s):
	        try:
	                d.append([float(s1.split(" ")[1]),float(s1.split(" ")[2]),float(s1.split(" ")[3])])
	                #d.append([float(s1.split(" ")[0]),float(s1.split(" ")[1]),float(s1.split(" ")[2])])

	        except:
	                continue
	d=array(d)

	# d[:,1][d[:,1]<0] = d[:,1][d[:,1]<0] + 2*pi

	err=circdist(d[:,1],d[:,2])
	#E.append(err)
	prev_curr = circdist(d[:,1],d[:,0])

	
	w1=pi/100
	w2=pi/3

	w1=pi/50

	xxx = arange(-pi,pi-w2+w1,w1)
	xxx = arange(0,pi-w2+w1,w1)
	err = sign(prev_curr)*err
	prev_curr=abs(prev_curr)
	prev_curr = prev_curr[abs(err)<radians(thr)]
	err=err[abs(err)<radians(thr)]
	
	m_err=[]
	std_err=[]
	count = []
	for x in xxx:
		idx=(prev_curr>x) & (prev_curr<=x+w2)
		m_err.append(circmean(err[idx],low=-pi,high=pi))
		std_err.append(circstd(err[idx],low=-pi,high=pi))
		count.append(sum(idx))

	xxx+=w2/2
	xxx=degrees(xxx)
	m_err = degrees(m_err)
	std_err = degrees(std_err)
	M.append(m_err)
	S.append(std_err/sqrt(count))
	E.append(err)
f=open("../preprocessed_data/sbias_model.pickle","w")

dump([xxx,M,S],f)
