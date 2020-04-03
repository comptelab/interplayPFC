from __future__ import division
import numpy as np
from scipy.stats import ttest_rel,ttest_1samp,ttest_ind,zscore
from matplotlib.pylab import *
from pickle import dump,load
from scikits import bootstrap
import seaborn as sns
from scipy.stats import *
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")


sns.set_context("poster", font_scale=0.8)
sns.set_style("ticks")

fig=figure(figsize=(6,10))

ax_beh  = subplot2grid((7, 4), (0, 2), rowspan=2,colspan=2)
ax_fr = subplot2grid((7, 4), (2, 0), rowspan=2,colspan=4)
ax_dec = subplot2grid((7,4), (4, 0), rowspan=2,colspan=4)
ax_tuning1 = subplot2grid((7,4), (6, 0))
ax_tuning2 = subplot2grid((7,4), (6, 1))
ax_tuning3 = subplot2grid((7,4), (6, 2))
ax_tuning4 = subplot2grid((7,4), (6, 3))
ax_beh.tick_params(direction='in')
ax_fr.tick_params(direction='in')
ax_dec.tick_params(direction='in')
ax_tuning1.tick_params(direction='in')
ax_tuning2.tick_params(direction='in')
ax_tuning3.tick_params(direction='in')
ax_tuning4.tick_params(direction='in')



def sig_bar(ax,sigs,axis,y,color):
        w=diff(axis)[0]
        for i in sigs:
        	ax.fill_between([axis[i],axis[i]+w],[y[0],y[0]],[y[1],y[1]],color=color)

time = loadtxt("preprocessed_data/decoder_time.dat")
Zscored = loadtxt("preprocessed_data/decoder_zscored.dat")

time -=1 # align to stimulus onset, causal window


lw=2
n_perms = 1000
W1=0.05
W2=0.25


f=open("preprocessed_data/firing_rate_fig2.pickle")
[xx,mean_fr,stderr,fixon] = load(f)
fixoff = -7.1+1+3

f=open("preprocessed_data/tuning_tcfix2.pickle")
TC_FIX2=load(f)

f=open("preprocessed_data/monkey_behavior.pickle")
[xxx2,down,up,w2,m_err,perms] = load(f) 


## MONKEY BEHAVIOR 

sig_bigger = array([mean(m < array(perms)[:,i]) for i,m in enumerate(m_err)])
sig_lower = array([mean(m > array(perms)[:,i]) for i,m in enumerate(m_err)])

ax_beh.fill_between(degrees(xxx2+w2/2),degrees(down),degrees(up),color="gray",alpha=0.25)
ax_beh.plot(degrees(xxx2+w2/2),degrees(m_err),"k-",lw=1.5)
ax_beh.plot([0,100],[0,0],"k--",alpha=0.5)
ax_beh.set_xlim(degrees(w2/2),100)

sig_bar(ax_beh,find(sig_lower<0.025),degrees(xxx2+w2/2),[1.95,2],"black")
sig_bar(ax_beh,find(sig_bigger<0.025),degrees(xxx2+w2/2),[1.95,2],"black")



ax_beh.set_ylim(-1,2)
ax_beh.set_yticks([-1,0,1,2])
ax_beh.set_xlabel("relative location of\n" r"previous trial $\theta_{d}$ ($^\circ$)",fontsize=12)
ax_beh.set_ylabel(r"error in current trial $\theta_{e}$ ($^\circ$)",fontsize=12)

#despine(ax_beh)

# FIRING RATE

ax_fr.plot(xx+W2,mean_fr,"k",lw=lw)
ax_fr.fill_between(xx+W2,mean_fr-2*stderr,mean_fr+2*stderr,color="gray",alpha=0.5)
ax_fr.plot(xx+W2,zeros(len(xx)),"k--",alpha=0.3, lw=1.5)
ax_fr.fill_between([0,0.5],-1,1,color="gray",alpha=0.2,label="cue")
ax_fr.fill_between([-7.1+0.5,-7.1+1],-1,1,color="gray",alpha=0.2,label="cue")
ax_fr.fill_between([fixoff,fixoff+.35],-1,1,color="gray",alpha=0.2)



ax_fr.plot(fixon,-0.3,"k^",ms=18)
ax_fr.plot(fixoff,-0.3,"k^",ms=18)
ax_fr.text(fixon+0.2,0.3,"ITI",fontsize=15)
ax_fr.text(fixon-0.55,-0.24,"fixation\n    on",fontsize=15)
ax_fr.text(fixoff-0.55,-0.24,"fixation\n    off",fontsize=15)
ax_fr.text(0.6,0.25,"current \n  delay",fontsize=15)
ax_fr.text(-4.2,0.25,"previous \n  delay",fontsize=15)


ax_fr.set_ylabel("normalized activity (z-score)")

ax_fr.set_yticks([-0.3,0,0.3])
ax_fr.set_xticks([])
ax_fr.set_xlim(-4.5,1.5)
ax_fr.set_ylim(-0.3,0.4)


#despine(ax_fr)

# DECODER

time +=0.5/2 # align to window center


m_zs = nanmean(Zscored[time<-3,:],0)
delay = time<-3

first = -1*Zscored[:,m_zs <= nanpercentile(Zscored[delay,:],33.3) ]
second = -1*Zscored[:,m_zs <median(m_zs)]
third = -1*Zscored[:,m_zs > nanpercentile(Zscored[delay,:],66.6) ]


stderr_first = amap(lambda x: bootstrap.ci(x,method="pi",alpha=0.32,n_samples=n_perms),first)
ci_first = amap(lambda x: bootstrap.ci(x,method="pi",n_samples=n_perms,statfunction=nanmean,alpha=0.05),first)
ci_first_0005 = amap(lambda x: bootstrap.ci(x,method="pi",n_samples=n_perms,statfunction=nanmean,alpha=0.005),first)



ax_dec.fill_between(time, ci_first[:,0],ci_first[:,1],alpha=0.2,color=sns.xkcd_rgb["pale red"],label="high-decoding delay")
ax_dec.plot(time, mean(first,1),color=sns.xkcd_rgb["pale red"],lw=2)
ax_dec.plot(time,zeros(len(time)),"k--",alpha=0.3, lw=1.5)
ax_dec.plot([-4+0.5/2,-4+0.5/2+0.3],[-0.15,-0.15],color=sns.xkcd_rgb["pale red"],lw=5)
ax_dec.plot([-2.8+0.5/2,-2.8+0.5/2+0.3],[-0.15,-0.15],color=sns.xkcd_rgb["greenish"],lw=5)
ax_dec.plot([-1.5+0.5/2,-1.5+0.5/2+0.3],[-0.15,-0.15],color=sns.xkcd_rgb["deep blue"],lw=5)
ax_dec.plot([-.5+0.5/2,-0.5+0.5/2+0.3],[-0.15,-0.15],color=sns.xkcd_rgb["orange"],lw=5)
ax_dec.plot(time, mean(third,1),lw=2,color="black",alpha=0.5,label="low-decoding delay")

sig_first = find(ci_first[:,0]>0)
sig_first_0005 = find(ci_first_0005[:,0]>0)



sig_bar(ax_dec,sig_first,time,[4.4,4.5],"gray")
sig_bar(ax_dec,sig_first_0005,time,[4.4,4.5],"black")


ax_dec.legend()

ax_dec.fill_between([0,0.5],-2,5,color="gray",alpha=0.2,label="cue")
ax_dec.fill_between([fixoff,fixoff+.35],-2,5,color="gray",alpha=0.2)


ax_dec.set_ylim([-1,4.5])
ax_dec.set_xlim([-4,1])
ax_dec.set_xlim(-4.5,1.5)
ax_dec.set_ylabel(r"distance from shuffle $(\sigma)$")
ax_dec.set_xlabel("time from stimulus (s)")


# tuning curves

t0=60
t1=72
t2=95
t3=85


stderr= amap(lambda x: bootstrap.ci(x,method="pi",statfunction=nanmean,alpha=0.32), TC_FIX2[t0,:,:].T)
mean_err=nanmean(TC_FIX2[t0,:,:],0)
mean_err = concatenate([[mean_err[-1]],mean_err])
stderr = concatenate([[stderr[-1]],stderr])

delay_m=mean_err
delay_s=stderr
test_delay = TC_FIX2[t0,:,3]
test_delay=ttest_1samp(test_delay,0,nan_policy="omit")


ax_tuning1.plot(linspace(-pi,pi,9),delay_m,color=sns.xkcd_rgb["pale red"],lw=lw)
ax_tuning1.fill_between(linspace(-pi,pi,9),delay_s[:,0],delay_s[:,1],alpha=0.2,color=sns.xkcd_rgb["pale red"])
ax_tuning1.plot([-pi,pi],[0,0],"k--",alpha=0.3, lw=1.5)
ax_tuning1.set_xticklabels(["preferred cue"])
ax_tuning1.set_xticks([0])
ax_tuning1.set_ylim(-1,1)
ax_tuning1.set_yticks([-1,0,1])
ax_tuning1.set_ylabel("tuning to \n previous stimulus")
ax_tuning1.tick_params(axis='y', direction="in")



stderr=nanstd(TC_FIX2[t1,:,:],0)/sqrt(sum(sum(isnan(TC_FIX2[t1,:,:]),1)<1))
stderr= amap(lambda x: bootstrap.ci(x,method="pi",statfunction=nanmean,alpha=0.32), TC_FIX2[t1,:,:].T)
mean_err=nanmean(TC_FIX2[t1,:,:],0)
mean_err = concatenate([[mean_err[-1]],mean_err])
stderr = concatenate([[stderr[-1]],stderr])
early_m=mean_err
early_s=stderr
test_early = TC_FIX2[t1,:,3]
test_early=ttest_1samp(test_early,0,nan_policy="omit")


ax_tuning2.plot(linspace(-pi,pi,9),early_m,color=sns.xkcd_rgb["greenish"],lw=lw)
ax_tuning2.fill_between(linspace(-pi,pi,9),early_s[:,0],early_s[:,1],alpha=0.2,color=sns.xkcd_rgb["greenish"])
ax_tuning2.plot([-pi,pi],[0,0],"k--",alpha=0.3, lw=1.5)
ax_tuning2.set_xticks([0])
ax_tuning2.set_xticklabels([])
ax_tuning2.set_yticklabels([])
ax_tuning2.tick_params(axis='y', direction="in")
ax_tuning2.yaxis.tick_right()
ax_tuning2.set_ylim(-0.3,0.3)
ax_tuning2.set_yticks([-0.3,0,0.3])


stderr=nanstd(TC_FIX2[t3,:,:],0)/sqrt(sum(sum(isnan(TC_FIX2[t3,:,:]),1)<1))
stderr= amap(lambda x: bootstrap.ci(x,method="pi",statfunction=nanmean,alpha=0.32), TC_FIX2[t3,:,:].T)
mean_err=nanmean(TC_FIX2[t3,:,:],0)
mean_err = concatenate([[mean_err[-1]],mean_err])
stderr = concatenate([[stderr[-1]],stderr])
late_m=mean_err
late_s =stderr
test_late = TC_FIX2[t3,:,3]
test_late=ttest_1samp(test_late,0,nan_policy="omit")


ax_tuning3.plot(linspace(-pi,pi,9),late_m,color=sns.xkcd_rgb["deep blue"],lw=lw)
ax_tuning3.fill_between(linspace(-pi,pi,9),late_s[:,0],late_s[:,1],alpha=0.2,color=sns.xkcd_rgb["deep blue"])
ax_tuning3.plot([-pi,pi],[0,0],"k--",alpha=0.3, lw=1.5)


ax_tuning3.set_xticks([0])
ax_tuning3.set_xticklabels([])
ax_tuning3.set_yticklabels([])
ax_tuning3.yaxis.tick_right()


ax_tuning3.tick_params(axis='y', direction="in")

ax_tuning3.set_ylim(-0.3,0.3)
ax_tuning3.set_yticks([-0.3,0,0.3])


stderr=nanstd(TC_FIX2[t2,:,:],0)/sqrt(sum(sum(isnan(TC_FIX2[t2,:,:]),1)<1))
stderr= amap(lambda x: bootstrap.ci(x,method="pi",statfunction=nanmean,alpha=0.32), TC_FIX2[t2,:,:].T)
mean_err=nanmean(TC_FIX2[t2,:,:],0)
mean_err = concatenate([[mean_err[-1]],mean_err])
stderr = concatenate([[stderr[-1]],stderr])
mid_m=mean_err
mid_s =stderr
test_mid = TC_FIX2[t2,:,3]
test_mid=ttest_1samp(test_mid,0,nan_policy="omit")


ax_tuning4.plot(linspace(-pi,pi,9),mid_m,color=sns.xkcd_rgb["orange"],lw=lw)
ax_tuning4.fill_between(linspace(-pi,pi,9),mid_s[:,0],mid_s[:,1],alpha=0.2,color=sns.xkcd_rgb["orange"])
ax_tuning4.plot([-pi,pi],[0,0],"k--",alpha=0.3, lw=1.5)
ax_tuning4.yaxis.tick_right()
ax_tuning4.tick_params(axis='y', direction="in")

ax_tuning4.set_xticks([0])
ax_tuning4.set_xticklabels([])
ax_tuning4.set_ylim(-0.3,0.3)
ax_tuning4.set_yticks([-0.3,0,0.3])

