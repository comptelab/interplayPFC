from __future__ import division
from matplotlib.mlab import find
from scipy.stats import *
from scikits import bootstrap
import seaborn as sns
from matplotlib.pylab import *
from numpy import *
import numpy as np
import bz2

set_printoptions(precision=4)
sns.set_context("talk", font_scale=1.4)

sns.set_style("ticks")


# beh vs dec
x=5
fig=figure(figsize=(3*x,2*x))


f = bz2.BZ2File('preprocessed_data/0.05_1.0_beh_vs_dec_others.pickle.bz2')
[time,ci_h,res,ps,serial,good_pairs,xxx,w2] = load(f,allow_pickle=True)
fixoff = -7.1+1+3


def perm_test(a,b):
	d = nanmean(a-b)
	perms=[]
	for n in range(10000):
		idx = random.random(len(a))<0.5
		a[idx],b[idx] = b[idx],a[idx]
		d_perm= nanmean(a-b)
		perms.append(d_perm)
	return mean(d<=array(perms))


def perm_test2(a):
	d = nanmean(a,1)
	perms=[]
	for n in range(10000):
		[shuffle(h) for h in a.T]
		d_perm= nanmean(a,1)
		perms.append(d_perm)
	return d,array(perms)


def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	continuity = diff(sigs)
	for i,c in enumerate(continuity):
		beg =axis[sigs[i]]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)



# SPLIT FOR 2 time points
silent=89
reactiv=111

for i,time_point in enumerate([silent, reactiv]):
	subplot(2,3,i+1)
	serial_low=serial[time_point,good_pairs,1].T
	serial_high=serial[time_point,good_pairs,0].T


	mean_serial_high = nanmean(serial_high,1)
	mean_serial_low = nanmean(serial_low,1)

	# bootstrapped stderr
	ci_low=amap(lambda x: bootstrap.ci(x,statfunction=nanmean,alpha=0.32),serial[time_point,good_pairs,1].T)
	ci_high=amap(lambda x: bootstrap.ci(x,statfunction=nanmean,alpha=0.32),serial[time_point,good_pairs,0].T)


	ps=[]
	for j in range(len(serial_low)):
		idx = ~isnan(serial_low[j]) & ~isnan(serial_high[j])
		p=perm_test(serial_high[j][idx],serial_low[j][idx])
		ps.append(p)

	d=perm_test2(serial_high)
	ps_high=mean(d[0]-d[1]<0,0)
	ps_low=mean(d[0]-d[1]>0,0)


	plot(xxx+w2/2,mean_serial_high,color=sns.xkcd_rgb["pale red"])
	fill_between(xxx+w2/2,ci_high[:,0],ci_high[:,1],color=sns.xkcd_rgb["pale red"],alpha=0.2,label="high-decoding trials")

	plot(xxx+w2/2,mean_serial_low,"-",color="black")
	fill_between(xxx+w2/2,ci_low[:,0],ci_low[:,1],color="gray",alpha=0.2,label="other trials")

	plot(xxx+w2/2,zeros(len(xxx)),"k--",alpha=0.3)
	sig_bar(find(array(ps)<0.05),xxx+w2/2,[0.95*3,3],"black")
	ylabel(r"error in current trial ($^\circ$)")
	xlim((xxx+w2/2)[0],(xxx+w2/2)[-1])
	locator_params(nbins=4)
	xticks([30,90,150])
	ylim(-3,3)
	xlim([(xxx+w2/2)[0],150])
	yticks(range(-3,4))
	sns.despine()
	legend()
	tick_params(direction='in')

#SPLIT FOR all time points
x=subplot(2,3,3)


high = amap(lambda x: percentile(x,97.5),ci_h.T)
low = amap(lambda x: percentile(x,2.5),ci_h.T)
ci = [low,high]


fill_between(time,array(ci)[0,:],array(ci)[1,:],alpha=0.5,color="gray", label="95% C.I.")
plot(time[silent],-0.75,"k^",ms=10)
plot(time[reactiv],-0.75,"^",ms=15,color="orange")
fill_between([0,0.5],[-5,-5],[10,10],alpha=0.2,color="gray")
fill_between([fixoff,fixoff+.35],[-5,-5],[10,10],color="gray",alpha=0.2)
plot(time,mean(res,0),color="black",lw=3)
plot(time,zeros(len(time)),"k--",alpha=0.3)

xticks(range(-4,2),)
xlim(-4,1.5)
ylim(-5,10)
yticks([-5,0,5,10])
ylabel(r"difference in serial bias ($^\circ$)")
legend()
sns.despine()
tick_params(direction='in')




# EEG
import pickle

with open('preprocessed_data/EEG_split.pkl', 'rb') as f:
    biases, time, split, high, low, tmax, tmin = pickle.load(f)


highmax = biases[:,:,0,tmax]
lowmax   = biases[:,:,1,tmax]
highcimax = np.array(list(map(lambda x: bootstrap.ci(x, alpha=.32), highmax.T))).T
lowcimax = np.array(list(map(lambda x: bootstrap.ci(x, alpha=.32), lowmax.T))).T


ps=[]
for j in range(lowmax.shape[1]):
    p=perm_test(np.copy(highmax[:,j]),np.copy(lowmax[:,j]))
    ps.append(p)

ps = np.array(ps)
sighighmax = np.where(ps<.05)[0]

highmin = biases[:,:,0,tmin]
lowmin   = biases[:,:,1,tmin]
highcimin = np.array(list(map(lambda x: bootstrap.ci(x, alpha=.32), highmin.T))).T
lowcimin = np.array(list(map(lambda x: bootstrap.ci(x, alpha=.32), lowmin.T))).T

ps=[]
for j in range(lowmin.shape[1]):
    p=perm_test(np.copy(highmin[:,j]),np.copy(lowmin[:,j]))
    ps.append(p)

ps = np.array(ps)
sighighmin = np.where(ps<.05)[0]

w1=pi/20
w2=pi/3

subplot(2,3,5)
xxx = arange(0, pi, w1) + w2/2
plot(xxx,np.mean(highmax, 0), sns.xkcd_rgb["pale red"], lw = 2)
plot(xxx,np.mean(lowmax, 0), 'k', lw = 2)
fill_between(xxx, highcimax[0], highcimax[1], color = sns.xkcd_rgb["pale red"],
alpha = .2, label = 'high decoding')
fill_between(xxx, lowcimax[0], lowcimax[1], color = 'gray',
alpha = .2, label = 'low decoding')
sig_bar(sighighmax,xxx,[radians(0.95*2),radians(2)],'black')
plot(xxx,np.zeros(len(xxx)),"k--",alpha=0.3)
xticks(np.deg2rad([30,90,150]),[30,90,150])
yticks(np.deg2rad([-2,-1,0,1,2]), [])
xlim([xxx[0],radians(150)])
ylim(np.deg2rad([-1,2]))
tick_params(direction='in')
sns.despine()

subplot(2,3,4)
plot(xxx,np.mean(highmin, 0), sns.xkcd_rgb["pale red"], lw = 2)
plot(xxx,np.mean(lowmin, 0), 'k', lw = 2)
fill_between(xxx, highcimin[0], highcimin[1], color = sns.xkcd_rgb["pale red"],
alpha = .2, label = 'high decoding')
fill_between(xxx, lowcimin[0], lowcimin[1], color = 'gray',
alpha = .2, label = 'low decoding')
sig_bar(sighighmin,xxx,(np.deg2rad(1.27),np.deg2rad(1.3)),'k')
plot(xxx,np.zeros(len(xxx)),"k--",alpha=0.3)
xticks(np.deg2rad([30,90,150]),[30,90,150])
yticks(np.deg2rad([-2,-1,0,1,2]), [-2,-1,0,1,2])
xlim([xxx[0],radians(150)])
ylim(np.deg2rad([-2,2]))
ylim(np.deg2rad([-1,2]))
ylabel('error in current trial ($^\circ$)')
xlabel('relative location in previous trial ($^\circ$)')
sns.despine()
tick_params(direction='in')

x=subplot(2,3,6)

plot(time, degrees(split), 'k', lw = 3)
fill_between(time, degrees(high), degrees(low), alpha=0.5,color="gray", label="95% C.I.")

fill_between([0,.25],[-1,-1],[5,5],alpha=0.2,color="gray")
plot(time,np.zeros(len(time)),"k--",alpha=0.3)
plot(time[tmax],-.15, '^', ms = 15,color="orange")
plot(time[tmin],-.15, 'k^', ms = 10)
xticks([-1,-.5,0, 0.5])
xlim([-1.25,.5])
xlabel('time from stimulus (s)')
ylabel(r'difference in absolute bias ($^\circ$)')

ylim([-1,3])
yticks(range(-1,4))
sns.despine()
tick_params(direction='in')

