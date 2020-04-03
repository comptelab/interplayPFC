from __future__ import division
import matplotlib.pyplot as plt
from numpy import *
from scipy.io import *
from scipy.stats import ttest_1samp,ttest_ind,ttest_rel
import pandas as pd
import sys
sys.path.insert(0, '../helpers/')
from helpers import *
from circ_stats import *
from constants import *
import statsmodels.formula.api as smf

set_printoptions(precision=4)
sns.set_context("talk", font_scale=0.9)
sns.set_style("ticks")
sns.set_style({"ytick.direction": "in"})

n_perms = 1000

PFC = 1
VERTEX = 0
LOW_TMS = 1 # only meaningful for PFC
HIGH_TMS = 2
NO_TMS = 0
BOTH_TMS = 1 # only meaningful for Vertex


replication = pd.read_csv("../Data/tms_ori_long_dataframe.csv")
original = pd.read_csv("../Data/tms_rep_long_dataframe.csv")


replication["study"] = 1
original["study"] = 0

replication["err"]=circdist(replication.report_angle.values,replication.target_angle.values)
original["err"]=circdist(original.report_angle.values,original.target_angle.values)

original["prev_curr"] = circdist(roll(original.target_angle.values,1), original.target_angle.values)
replication["prev_curr"] = circdist(roll(replication.target_angle.values,1), replication.target_angle.values)

df = pd.concat((replication,original))

df["fold_error"] = df.err * sign(df.prev_curr)
df["tms_abs"] = df.RMT * df.tms_intensity
df["prev_tms"] = roll(df.tms_intensity,1)

# remove random guesses
df=df[df.err.abs()<radians(45)]
df_model = df.copy()


# save for R analyses
df_model.to_csv("R_modeling/data_for_R_model_raw_tms.csv", sep='\t')

# Transform TMS high into -1, making it linear relationship with sbias
# # This a prediction from the model: strong TMS should decrease serial biases
df_model.tms_intensity[df_model.tms_intensity > 1] = -1
df_model.tms_intensity[df_model.tms_intensity > 0] = 1

# save for R analyses
df_model.to_csv("R_modeling/data_for_R_model.csv", sep='\t')


## COMPUTE SERIAL BIAS
tms_intensities = df.tms_intensity.unique()
locations = df.location.unique()
subjects = df.subject.unique()

allsbias_all = zeros([len(subjects),len(locations),len(tms_intensities),len(xxx)])
allsbias_first = zeros([len(subjects),len(locations),len(tms_intensities),len(xxx)])
allsbias_last = zeros([len(subjects),len(locations),len(tms_intensities),len(xxx)])

for sub_i,(sub, subject) in enumerate(df.groupby("subject")):
	for loc_i, (loc,location) in enumerate(subject.groupby("location")):
		for tms_i, (tms,tms_int) in enumerate(location.groupby("tms_intensity")):
			sbias = zeros(len(xxx))
			sbias_first = zeros(len(xxx))
			sbias_last = zeros(len(xxx))
			n_sessions = len(tms_int.session.unique())
			for sess_i, (_,session) in enumerate(tms_int.groupby("session")):
				sbias_all_first = compute_seria_from_pandas(session[:int(len(session)/2)],xxx,flip)
				sbias_all_last = compute_seria_from_pandas(session[int(len(session)/2):],xxx,flip)
				sbias_all = compute_seria_from_pandas(session,xxx,flip)
				sbias += sbias_all[2]
				sbias_first += sbias_all_first[2]
				sbias_last += sbias_all_last[2]

			allsbias_all[sub_i,loc_i,tms_i] = sbias/n_sessions
			allsbias_first[sub_i,loc_i,tms_i] = sbias_first/n_sessions
			allsbias_last[sub_i,loc_i,tms_i] = sbias_last/n_sessions

## PLOT SERIAL BIAS
figure(figsize=(10,10))

titles=["full session", "first half", "last half"]
for i,allsbias in enumerate([allsbias_all, allsbias_first,allsbias_last]):
#for i,allsbias in enumerate([allsbias_first]):

	subplot(3,3,3*i+1)
	title("Vertex")
	plot_serial(allsbias[:,VERTEX,NO_TMS,:],"k",label="sham")
	plot_serial(allsbias[:,VERTEX,BOTH_TMS,:],sns.xkcd_rgb["greenish"],label="weak tms")
	plot_serial(allsbias[:10,VERTEX,BOTH_TMS,:],"darkblue",label="strong tms")
	yticks([-2,-1,0,1,2])
	ylabel(r"error in current trial ($^\circ$)")
	xlabel("")
	if i!=2:
		xticks(xticks()[0],"")
	else:
		legend(loc='lower right')
	ylim(-2,2)
	xlim(xxx2[0],150)


	subplot(3,3,3*i+2)
	title("PFC")
	plot_serial(allsbias[:,PFC,NO_TMS,:],"k")
	plot_serial(allsbias[:,PFC,LOW_TMS,:],sns.xkcd_rgb["greenish"],label="weak tms")
	plot_serial(allsbias[:,PFC,HIGH_TMS,:],"darkblue",label="strong tms")

	ylabel("")
	xlabel("")
	yticks([-2,-1,0,1,2])
	if i!=2:
		xticks(xticks()[0],"")
	yticks(yticks()[0],"")
	ylim(-2,2)
	xlim(xxx2[0],150)

	subplot(3,3,3*i+3)
	d_pfc_l=allsbias[:,PFC,LOW_TMS,:] - allsbias[:,PFC,NO_TMS,:]
	d_vertex=allsbias[:,VERTEX,BOTH_TMS,:] - allsbias[:,VERTEX,NO_TMS,:]
	pvalues = array([boot_test(d_pfc_l[:,p] - d_vertex[:,p],0, n_samples=n_perms)[0] for p in range(shape(allsbias)[3])])/2
	plot_serial(d_pfc_l,"r","PFC")
	plot_serial(d_vertex,"k","Vertex")
	plot_sigs(d_pfc_l,"k",d_vertex,pvalues=pvalues,upper=[1.9,2])
	ylim(-2,2)
	yticks([-2,-1,0,1,2])
	yticks(yticks()[0],"")
	xlabel(r"relative location of previous trial ($^\circ$)")
	ylabel("error in current trial\n" r"relative to sham ($^\circ$)")
	if i!=2:
		xlabel("")
		xticks(xticks()[0],"")
	else:
		legend(loc='lower right')
	xlim(xxx2[0],150)

