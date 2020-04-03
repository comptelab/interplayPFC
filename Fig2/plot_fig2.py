import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pickle
from scikits.bootstrap import bootstrap as boot
import seaborn as sns

##############################################################################
#				FUNCTIONS FOR PLOTTING AND STATISTICAL TESTS	             #
##############################################################################

def sig_bar(ax,sigs,axis,y,color):
	w=np.diff(axis)[0]
	continuity=np.diff(sigs)
	for i,c in enumerate(continuity):
		beg = axis[sigs[i]]-w/2
		end = beg+w
		ax.fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)

def boot_test(data, thr=0, n_samples=1000000):
	data 		= np.array(data)	
	t_data  	= np.nanmean(data) - thr
	boot_data 	= data[boot.bootstrap_indexes_array(data, n_samples=n_samples)]
	t_boot 		= (np.nanmean(boot_data,1) - np.nanmean(data))
	p 			=  np.nanmean(abs(t_data)<=abs(t_boot))
	return p, np.percentile(np.mean(boot_data,1),[2.5,97.5])


##############################################################################
#							LOAD DATA FOR PLOTTING	  		                 #
##############################################################################

with open('preprocessed_data/cross_decoders.pkl', 'rb') as f: 
    dtime, delay, resp, sig_005, t_delay = pickle.load(f)

with open('preprocessed_data/crossmatrix_decoders.pkl', 'rb') as f:
    mtime, matrix = pickle.load(f)

with open('preprocessed_data/serial_bias.pkl', 'rb') as f:
    xxx, serial, sigpos, signeg = pickle.load(f)


##############################################################################
#							SETTINGS FOR PLOTTING	  		                 #
##############################################################################

np.set_printoptions(precision=4)
sns.set_context("talk", font_scale=1)
sns.set_style("ticks")

orange 	= sns.xkcd_rgb["orange"]
greenish = sns.xkcd_rgb["greenish"]
deepblue = sns.xkcd_rgb["deep blue"]
palered = sns.xkcd_rgb["pale red"]


##############################################################################
#							SET UP FIGURE LAYOUT	  		                 #
##############################################################################

plt.figure(figsize=(9,8.5))

gs1 = gs.GridSpec(3, 7)
gs1.update(bottom=.56, wspace=1)
ax1 = plt.subplot(gs1[:,:3])
ax2 = plt.subplot(gs1[:,3:6])
ax22 = plt.subplot(gs1[:,-1])

gs2	= gs.GridSpec(1,3, width_ratios=[2, 1, 3])
gs2.update(top=.45, right=.7)
ax6 = plt.subplot(gs2[0])
ax7 = plt.subplot(gs2[1])
ax8 = plt.subplot(gs2[2])

gs3 = gs.GridSpec(3, 1)
gs3.update(top=.45, left=.8)
ax3 = plt.subplot(gs3[0,-1])
ax4 = plt.subplot(gs3[1,-1])
ax5 = plt.subplot(gs3[2,-1])


##############################################################################
#						PLOT DELAY AND RESPONSE DECODER	  		   	         #
##############################################################################

late 	= np.where((dtime[1,:]==0) & (dtime[0,:]>.65) & (dtime[0,:]<.85))[0]
click 	= np.where((dtime[1,:]==2) & (dtime[0,:]>-.1) & (dtime[0,:]<.1))[0]
reig 	= np.where((dtime[1,:]==3) & (dtime[0,:]>-.95) & (dtime[0,:]<-.75))[0]

nsubs 	= delay.shape[0]

times 	= np.where((dtime[1,:]==0) & (dtime[0,:]>.25) & (dtime[0,:]<1.5))[0]
S 		= np.where((dtime[1,:]==3) & (dtime[0,:]>0) & (dtime[0,:]<.25))[0]
ax6.plot(dtime[0,times], np.mean(delay,0)[times], color = 'k')
ax6.fill_between(dtime[0,times], np.mean(delay,0)[times] + 2*sps.sem(delay,0)[times],
	np.mean(delay,0)[times] - 2*sps.sem(delay,0)[times], color='grey', alpha=.5)
ax6.plot(dtime[0,times], np.mean(resp,0)[times],'k', alpha=.5)
ax6.plot(dtime[0,times], np.zeros(len(times)), 'k--', alpha=.3)
ax6.plot(dtime[0,late], np.zeros(len(dtime[0,late]))-.015, 'o', color=palered)
sig_bar(ax6,np.where(sig_005[times])[0],dtime[0,times],[.19,.2], 'k')
ax6.plot(dtime[0,S[0]], -.195, 'k^', ms=14)
ax6.set_xlim([.25,1.25])
ax6.set_xticks([.5,1])
ax6.set_ylim([-.2,.2])
ax6.set_yticks([-.2,-.1,0,.1,.2])
ax6.set_ylabel('decoding strength (a.u.)')
ax6.get_yaxis().set_tick_params(direction='in')
ax6.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax6)

times 	= np.where((dtime[1,:]==2) & (dtime[0,:]>-.3) & (dtime[0,:]<.3))[0]
R 		= np.where((dtime[1,:]==2) & (dtime[0,:]>0))[0][0]
ax7.plot(dtime[0,times], np.mean(delay,0)[times], color='k')
ax7.fill_between(dtime[0,times], np.mean(delay,0)[times] + 2*sps.sem(delay,0)[times],
	np.mean(delay,0)[times] - 2*sps.sem(delay,0)[times], color='grey', alpha=.5)
ax7.plot(dtime[0,times], np.mean(resp,0)[times], 'k', alpha=.5)
ax7.plot(dtime[0,times], np.zeros(len(times)), 'k--', alpha=.3)
ax7.plot(dtime[0,click], np.zeros(len(dtime[0,click]))-.015, 'o', color=deepblue)
sig_bar(ax7,np.where(sig_005[times])[0],dtime[0,times],[.19,.2], 'k')
ax7.plot(dtime[0,R], -.195, 'k^', ms=14)
ax7.text(-.23, -.17, r'$(R_{n-1})$') 
ax7.text(-.23, -.13, 'report') 
ax7.set_ylim([-.2,.2])
ax7.set_xlim([-.25,.25])
ax7.set_xticks([-.25,.25])
ax7.set_xlabel('time from point of alignment (s)')
ax7.set_yticks([])
ax7.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax7, left=True)

times 	= np.where((dtime[1,:]==3) & (dtime[0,:]>-1.5) & (dtime[0,:]<.5))[0]
F 		= np.where((dtime[1,:]==3) & (dtime[0,:]>-1.1))[0][0]
S 		= np.where((dtime[1,:]==3) & (dtime[0,:]>0) & (dtime[0,:]<.25))[0]
ax8.plot(dtime[0,times], np.mean(delay,0)[times], color='k')
ax8.fill_between(dtime[0,times], np.mean(delay,0)[times] + 2*sps.sem(delay,0)[times],
	np.mean(delay,0)[times] - 2*sps.sem(delay,0)[times], color='grey', alpha=.5,
	label='delay code')
ax8.plot(dtime[0,times], np.mean(resp,0)[times], 'k', alpha=.5, 
	label='response code')
ax8.plot(dtime[0,times], np.zeros(len(times)), 'k--', alpha=.3)
ax8.fill_between(dtime[0,S], np.zeros(len(S))-.2, np.zeros(len(S))+.2, 
	color='grey', alpha=.2)
ax8.plot(dtime[0,reig], np.zeros(len(dtime[0,reig]))-.015, 'o', color=orange)
sig_bar(ax8,np.where(sig_005[times])[0],dtime[0,times],[.19,.2], 'k')
ax8.plot(dtime[0,F], -.195, 'k^', ms=14)
ax8.plot(dtime[0,S[0]], -.195, 'k^', ms=14)
ax8.text(-1.25, -.13, 'fixation on')
ax8.text(-.3, -.13, 'stim on')
ax8.text(-1.250, -.17, r'$(F_n)$') 
ax8.text(-.1, -.17, r'$(S_n)$')
ax8.set_ylim([-.2,.2])
ax8.set_xlim([-1.25,.25])
ax8.set_xticks([-1,-.5,0])
ax8.set_yticks([-.2,-.1,0,.1,.2]); ax8.set_yticklabels([])
sns.despine(ax=ax8, left=True, right=False)
ax8.legend(frameon=False)
ax8.get_yaxis().set_tick_params(direction='in')
ax8.get_xaxis().set_tick_params(direction='in')


##############################################################################
#							PLOT DELAY TUNING CURVES	  		   	         #
##############################################################################

tun 	= np.roll(np.mean(np.mean(t_delay,0)[:,late],1),4)
tun 	= np.append(tun, tun[0]) - np.mean(tun)
se 		= np.roll(np.std(np.mean(t_delay[:,:,late],2),0),4)/np.sqrt(nsubs)
se 		= np.append(se, se[0])
ax3.plot(np.arange(9), tun, color=palered)
ax3.fill_between(np.arange(9), tun+se, tun-se, color=palered, alpha=.3)
ax3.plot(np.arange(9), np.zeros(9), 'k--', alpha=.3)
ax3.set_ylim([-.5,.5])
ax3.set_xlim([0,8])
ax3.set_yticks([-.5,0,.5]); ax3.set_yticklabels([])
sns.despine(ax=ax3, left=True, right=False)
ax3.get_yaxis().set_tick_params(direction='in')
ax3.get_xaxis().set_tick_params(direction='in')
ax3.set_xticks([4]); ax3.set_xticklabels([''])

print(boot_test(np.mean(t_delay[:,:,late],2)[:,0]-np.mean(t_delay[:,:,late],2)[:,4], n_samples=10000000))

tun 	= np.roll(np.mean(np.mean(t_delay,0)[:,click],1),4)
tun 	= np.append(tun, tun[0]) - np.mean(tun)
se 		= np.roll(np.std(np.mean(t_delay[:,:,click],2),0),4)/np.sqrt(nsubs)
se 		= np.append(se, se[0])
ax4.plot(np.arange(9), tun, color=deepblue)
ax4.fill_between(np.arange(9), tun+se, tun-se, color=deepblue, alpha=.3)
ax4.plot(np.arange(9), np.zeros(9), 'k--', alpha=.3)
ax4.set_ylim([-.5,.5])
ax4.set_xlim([0,8])
ax4.set_yticks([-.5,0,.5])
ax4.set_ylabel('tuning to previous stimulus')
sns.despine(ax=ax4, left=True, right=False)
ax4.get_yaxis().set_tick_params(direction='in')
ax4.get_xaxis().set_tick_params(direction='in')
ax4.yaxis.set_label_position("right")
ax4.set_xticks([4]); ax4.set_xticklabels([''])

print(boot_test(np.mean(t_delay[:,:,click],2)[:,0]-np.mean(t_delay[:,:,click],2)[:,4], n_samples=10000000))

tun 	= np.roll(np.mean(np.mean(t_delay,0)[:,reig],1),4)
tun 	= np.append(tun, tun[0]) - np.mean(tun)
se 		= np.roll(np.std(np.mean(t_delay[:,:,reig],2),0),4)/np.sqrt(nsubs)
se 		= np.append(se, se[0])
ax5.plot(np.arange(9), tun, color=orange)
ax5.fill_between(np.arange(9), tun+se, tun-se, color=orange, alpha=.3)
ax5.plot(np.arange(9), np.zeros(9), 'k--', alpha=.3)
ax5.set_ylim([-.5,.5])
ax5.set_xlim([0,8])
ax5.set_yticks([-.5,0,.5]); ax5.set_yticklabels([])
ax5.set_xticks([0,4,8]); ax5.set_xticklabels([-180,0,180])
sns.despine(ax=ax5, left=True, right=False)
ax5.get_yaxis().set_tick_params(direction='in')
ax5.get_xaxis().set_tick_params(direction='in')
ax5.set_xlabel('presented cue')

print(boot_test(np.mean(t_delay[:,:,reig],2)[:,0]-np.mean(t_delay[:,:,reig],2)[:,4], n_samples=10000000))

##############################################################################
#					PLOT CROSS-TEMPORAL DECODING MATRIX	  		   	         #
##############################################################################

valid = np.where(((mtime[1,:]==0) & (mtime[0,:]>-.3) & (mtime[0,:]<1.3))| 
		((mtime[1,:]==2) & (mtime[0,:]>-.3) & (mtime[0,:]<.3)) | 
		((mtime[1,:]==3) & (mtime[0,:]>-1.3) & (mtime[0,:]<.3) ))[0]

mat    = matrix[:,valid,:][:,:,valid]
mt	   = mtime[:,valid]

cut1   = np.where((mt[1,:]==0))[0][-1]
cut2   = np.where((mt[1,:]==2))[0][-1]

stim   = np.where((mt[0,:]>0))[0][0]
delay  = np.where((mt[0,:]>0.75))[0][0]
resp   = np.where((mt[0,:]<0) & (mt[1,:]==2))[0][-1]
fix    = np.where((mt[0,:]<-1.1) & (mt[1,:]==3))[0][-1]
stim2  = np.where((mt[0,:]>0) & (mt[1,:]==3))[0][0]

nsamps = mat.shape[1]

c = ax2.imshow(np.mean(mat,0), aspect='auto', origin='lower', 
	vmin=0, vmax=.1, cmap='plasma')
ax2.plot([cut1,cut1], [-.3,nsamps-.5], 'w-', lw=2)
ax2.plot([cut2,cut2], [-.3,nsamps-.5], 'w-', lw=2)
ax2.plot([-.3,nsamps-.5], [resp,resp], 'w--', lw=2)
ax2.plot([-.3,nsamps-.5], [delay,delay], 'w--', lw=2)
ax2.plot([-.3,nsamps-.5], [cut1,cut1], 'w-', lw=2)
ax2.plot([-.3,nsamps-.5], [cut2,cut2], 'w-', lw=2)
ax2.fill_between([nsamps-5.5,nsamps-.5],[5.5,5.5],[4,4], color='k')
ax2.text(nsamps-10, 3, '.5 s')
ax2.set_yticks([stim, resp, fix, stim2]); 
ax2.set_yticklabels([r'$S_{n-1}$', r'$R_{n-1}$', r'$F_n$', r'$S_n$'], 
	rotation='vertical')
ax2.set_xticks([stim, resp, fix, stim2]);  
ax2.set_xticklabels([r'$S_{n-1}$', r'$R_{n-1}$', r'$F_n$', r'$S_n$'])
ax2.set_xlabel('testing time')
ax2.set_ylabel('training time')
ax2.set_ylim()
sns.despine(ax=ax2, left=True, bottom=True)
plt.colorbar(mappable=c, ax=ax22, shrink=2, ticks=[0,.1], 
	label='decoding strength (a.u.)')
sns.despine(ax=ax22, left=True, bottom=True)
ax22.set_xticks([]);  ax22.set_yticks([]);  


##############################################################################
#								PLOT SERIAL BIAS	  			   	         #
##############################################################################

ax1.plot(xxx, np.nanmean(serial,0), color='k')
ax1.fill_between(xxx, np.mean(serial,0) + sps.sem(serial,0), 
	np.nanmean(serial,0) - sps.sem(serial,0), color='grey', alpha=.25)
sig_bar(ax1,np.where(sigpos)[0],xxx,[.97,1], 'k')
sig_bar(ax1,np.where(signeg)[0],xxx,[.97,1], 'k')
ax1.plot(xxx, np.zeros(len(xxx)),'k--', alpha=.3)
ax1.set_ylim([-.5,1])
ax1.set_xlim([0,120])
ax1.set_ylabel(r'error in current trial $\theta^{\prime}_{e}$ ($^\circ$)')
ax1.set_yticks([-.5,0,.5,1])
ax1.set_xticks([0,60,120]); ax1.set_xticklabels([30,90,150])
ax1.set_xlabel('relative location of\n previous trial ' + r'$\theta_{d}$ ($^\circ$)')
ax1.get_yaxis().set_tick_params(direction='in')
ax1.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax1)

plt.show()
