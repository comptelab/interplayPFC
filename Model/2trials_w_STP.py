
# coding: utf-8

# ## A model of spatial working memory from Compte et al. (2000) implemented in Brian
# 
# Courtesy of David Andrieux

from __future__ import division
from brian import *
from numpy.fft import rfft,irfft
from scipy.special import erf
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from circ_stats import *
import random
import os
import time
import socket

log_file = "simulation_%i_%f_%s" %(os.getpid(), time.time(), socket.gethostname())
my_file = Path(log_file+"_spikes.dat")



i=0
while my_file.is_file():
    i+=1
    my_file = Path(log_file+str(i)+"_spikes.dat")

log_file+=str(i)

print log_file

# size of the network
NE          = 2048 #NE              # number of E cells 
NI          = 512 #256               # number of I cells 

# simulation-related parameters  
dt          = 0.02*ms     	# simulation step length [ms]
simulation_clock=Clock(dt=dt)   # clock for the time steps


# cue parameters
tc_start      = 500*ms           # start of the cue signal
tc_dur        =250*ms            # length period of the cue signal
i_cue_amp     = 0.25*nA#0.35*nA #0.35*nA#0.4#0.5*nA#0.2*nA            # cue current amplitude


i_cue_ang     = random.randint(0,360)#180         # mean angle of the cue current
i_cue_ang     = 180         # mean angle of the cue current
i_cue_width   = 20#15#20#40#14.4              # sigma/width of the current signal in degrees
dt_curr       = 50*ms             # current_clock step length (should be the highest common denominator between tc_start and tr_stop)
t_delay	      = 1000*ms		# delay duration

bkgs = array([0,3])
bkg_fact = bkgs[randint(2)]
drive = 1
if bkg_fact == 0:
    drive = 0


reset_start = tc_start+tc_dur+t_delay
reset_stop = reset_start+300*ms

restart_start=reset_stop+1000*ms
restart_stop = restart_start+300*ms

sec_cue_start=restart_stop
sec_cue_stop = sec_cue_start+tc_dur
sec_cue_ang = random.randint(0,359)
sec_cue_neuron = int(sec_cue_ang*NE/360)



fact_ee =0.6 #0.6 #0.61 #0.585 
fact_gext_e= 0.91 #0.9185 #0.925 #0.925 #0.93 #0.95 #0.9642 #0.96415 #0.9642 #0.96425
fact_ie = 0.98 
fact_ei = 1.

simtime     =sec_cue_stop + t_delay
# pyramidal cells
Cm_e        = 0.5*nF        # [nF] total capacitance
gl_e        = 25.0*nS       # [ns] total leak conductance
El_e        = -70.0*mV      # [mV] leak reversal potential
Vth_e       = -50.0*mV      # [mV] threshold potential
Vr_e        = -60.0*mV      # [mV] resting potential
tr_e        = 2.0*ms        # [ms] refractory time

# interneuron cells
Cm_i        = 0.2*nF        # [nF] total capacitance
gl_i        = 20.0*nS       # [ns] total leak conductance
El_i        = -70.0*mV      # [mV] leak reversal potential
Vth_i       = -50.0*mV      # [mV] threshold potential
Vr_i        = -60.0*mV      # [mV] resting potential
tr_i        = 1.0*ms        # [ms] refractory time

# external background input
fext_e      = 1800.0*Hz*ones(NE)        # [Hz] external input frequency (poisson train)
fext_i      = 1800.0*Hz*ones(NI)
fext_e      = 1900.0*Hz*ones(NE)        # [Hz] external input frequency (poisson train)
fext_i      = 1900.0*Hz*ones(NI)
fext=append(fext_e,fext_i)

# AMPA receptor (APMAR)
E_ampa      = 0.0*mV        # [mV] synaptic reversial potential
t_ampa      = 2.0*ms        # [ms] exponential decay time constant 
g_ext_e     = fact_gext_e*6.5*nS#0.975*3.1*nS       # [nS] maximum conductance from external to pyramidal cells
g_ext_i     = 5.8*nS        # [nS] maximum conductance from external to interneuron cells
Gee_ampa    = 0.1*nS #0.251*nS  #0.251*nS #0.391*nS      # [nS] maximum conductance from pyramidal to pyramidal cells
Gei_ampa    = 0.192*nS  #0.192*nS #0.293*nS      # [nS] maximum conductance from pyramidal to inhbitory cells

# GABA receptor (GABAR)
E_gaba      = -70.0*mV      # [mV] synaptic reversial potential
t_gaba      = 10.0*ms       # [ms] exponential decay time constant 
Gie     = fact_ie*0.935*nS          # [ns] synaptic conductance interneuron to pyramidal cells
Gii     = 0.7413*nS          # [ns] synaptic conductance interneuron to interneuron cells

# NMDA receptor (NMDAR)
E_nmda      = 0.0*mV        # [mV] synaptic reversial potential
t_nmda      = 100.0*ms      # [ms] decay time of NMDA currents
t_x     = 2.0*ms        # [ms] controls the rise time of NMDAR channels
alpha       = 0.5*kHz       # [kHz] controls the saturation properties of NMDAR channels
a           = 0.062/mV          # [1/mV] control the voltage dependance of NMDAR channel
b           = 1.0/3.57          # [1] control the voltage dependance of NMDAR channel ([Mg2+]=1mM )
Gee     = fact_ee*0.7*nS #0.381*nS    #0.274*nS      # [ns] synaptic conductance from pyramidal to pyramidal cells
Gei     = fact_ei*0.49*nS #0.292*nS   #0.212*nS       # [ns] synaptic conductance from pyramidal to interneuron cells

# Connection parameters
Jp_ee       = 6.95# 6.7 #6.75 #7 #6.5 #6.3       # maximum connection weight from pyramidal to pyramidal
Jp_ee = 7.1 # 7.075
sigma_ee    = 18. #9.4       # width of the connectivity footprint from pyramidal to pyramidal
sigma_ei =  32.4
sigma_ie =  32.4

Jp_ei = 2.2
Jp_ie = 1.89 #1.875 #1.8
Jp_ii = 1.4 #1.5


# short-term plasticity parameters
t_sx = 200 *ms
t_su = 1500*ms

U = 0.2
r0 = 2*Hz

# Synaptic scalings:
Gie = Gie/NI*512
Gii = Gii/NI*512
Gee = Gee/NE*2048
Gei = Gei/NE*2048
Gee_ampa = Gee_ampa/NE*2048
Gei_ampa = Gei_ampa/NE*2048
wrec_e=Gee_ampa/g_ext_e
wrec_i=Gei_ampa/g_ext_i


# connectivity
# connectivity footprint
tmp   = sqrt(2*pi)*sigma_ee*erf(360.*0.5/sqrt(2.)/sigma_ee)/360.
Jm_ee = (1.-Jp_ee*tmp)/(1.-tmp)

tmp   = sqrt(2*pi)*sigma_ie*erf(360.*0.5/sqrt(2.)/sigma_ie)/360.
Jm_ie = (1.-Jp_ie*tmp)/(1.-tmp)

tmp   = sqrt(2*pi)*sigma_ei*erf(360.*0.5/sqrt(2.)/sigma_ei)/360.
Jm_ei = (1.-Jp_ei*tmp)/(1.-tmp)


weight=lambda i:(Jm_ee+(Jp_ee-Jm_ee)*exp(-0.5*(360.*min(i,NE-i)/NE)**2/sigma_ee**2))

weight_e=zeros(NE)
for i in xrange(NE): 
    weight_e[i]=weight(i)

fweight = rfft(weight_e) # Fourier transform

weight_ii=lambda i:(Jm_ei+(Jp_ei-Jm_ei)*exp(-0.5*(360.*min(i,NE-i)/NE)**2/sigma_ei**2))

weight_i=zeros(NE)
for i in xrange(NE): 
    weight_i[i]=weight_ii(i)

fweight_i = rfft(weight_i) # Fourier transform


del weight_e


# function for circular boundary conditions

# return the normalised "distance" of the neurons in a circle
def circ_distance(fi):
    if (fi > 0):
        return min(fi,360-fi)
    else:
        return max(fi,360+fi)

def circ_distance(i,j):
    return degrees(circdist(i/NE*2*pi,j/NE*2*pi))

def ang2ne(ang):
    ne = ang/(2*pi)*NE
    if ne < 0:
        ne+=NE
    return ne

def ne2ang(ne):
    return ne/NE*(2*pi)

# define external stimulus details:

tc_stop       = tc_start + tc_dur
current_clock = Clock(dt=dt_curr)

currents = lambda i,j: i_cue_amp*exp(-0.5*circ_distance(i,j)**2/i_cue_width**2)

current_e=zeros(NE)
j = i_cue_ang*NE/360.
for i in xrange(NE): 
    current_e[i]=currents(i,j)

current_e2=zeros(NE)
j = sec_cue_ang*NE/360.
for i in xrange(NE): 
    current_e2[i]=currents(i,j)


# Neuron equations:
eqs_e = '''
dv/dt = (-gl_e*(v-El_e)-g_ext_e*s_ampa*(v-E_ampa)-Gee*s_tot*(v-E_nmda)/(1+b*exp(-a*v))-Gie*s_gaba*(v-E_gaba)+i_e)/Cm_e: volt 
ds_ampa/dt = -s_ampa/t_ampa : 1
ds_gaba/dt = -s_gaba/t_gaba : 1
ds_nmda/dt = -s_nmda/t_nmda+alpha*x*(1-s_nmda) : 1
dx/dt = -x/t_x : 1
s_tot : 1
i_e : amp
dsynx/dt = (1-synx)/t_sx :1
dsynu/dt = (U-synu)/t_su :1
synf :1
'''

eqs_i = '''
dv/dt = (-gl_i*(v-El_i)-g_ext_i*s_ampa*(v-E_ampa)-Gei*(v-E_nmda)*s_tot/(1+b*exp(-a*v))-Gii*s_gaba*(v-E_gaba))/Cm_i: volt 
ds_ampa/dt = -s_ampa/t_ampa : 1
ds_gaba/dt = -s_gaba/t_gaba : 1
s_tot : 1
'''



# Setting up the populations:

Pe = NeuronGroup(NE, eqs_e, threshold=Vth_e, reset="V=Vr_e;x+=synf/U;synx-=synf;synu+=U*(1-synu)", refractory=tr_e, clock=simulation_clock, order=2, freeze=True)
Pe.v = El_e
Pe.s_ext = 0
Pe.s_gaba = 0
Pe.s_nmda = 0
Pe.x = 0
Pe.i_e = 0
Pe.synu = 0.255
Pe.synx = 0.985

Pi = NeuronGroup(NI, eqs_i, threshold=Vth_i, reset=Vr_i, refractory=tr_i, clock=simulation_clock, order=2, freeze=True)
Pi.v = El_i
Pi.s_ext = 0
Pi.s_gaba = 0

# external Poisson input
PG = PoissonGroup(NE+NI, fext, clock=simulation_clock)
PGe = PG.subgroup(NE)
PGi = PG.subgroup(NI)


# Create the connections:

Cpe = IdentityConnection(PGe, Pe, 's_ampa', weight=1.0)
Cpi = IdentityConnection(PGi, Pi, 's_ampa', weight=1.0)

Cie = Connection(Pi,Pe,'s_gaba',weight=lambda i,j:(Jm_ie+(Jp_ie-Jm_ie)*exp(-0.5*(360.*min(abs(4*i-j),NE-abs(4*i-j))/NE)**2/sigma_ie**2)))
Cii = Connection(Pi, Pi, 's_gaba', weight=1.0)

Cei_ampa = Connection(Pe,Pi,'s_ampa',weight=lambda i,j:wrec_i*(Jm_ei+(Jp_ei-Jm_ei)*exp(-0.5*(360.*min(abs(i-4*j),NE-abs(i-4*j))/NE)**2/sigma_ei**2)),modulation='synf')
Cee_ampa = Connection(Pe,Pe,'s_ampa',weight=lambda i,j:wrec_e*(Jm_ee+(Jp_ee-Jm_ee)*exp(-0.5*(360.*min(abs(i-j),NE-abs(i-j))/NE)**2/sigma_ee**2)), modulation='synf')

# NMDA calculation to be performed at each time step:

# Calculate NMDA contributions
@network_operation(simulation_clock, when='start')
def update_nmda(simulation_clock):
    Pe.synf = Pe.synx * Pe.synu
    fsnmda = rfft(Pe.s_nmda) 
    fstot = fsnmda*fweight
    fstot_i = fsnmda*fweight_i
    Pe.s_tot = irfft(fstot, NE)
    Pi.s_tot = irfft(fstot_i,NE)[0:-1:4]

def time_cond(time,beg,end):
    return (time >= beg) & (time < end)

bkg=0.58e-10
bkg2=2.1*bkg
bkg2=bkg_fact*bkg

#~[0-1]
def smooth(t0,tcurr,duration):
    def f(x): 
        return 1-1/exp(3*x)
    # 1 when tcurr = t0, 0 when starting
    x = (tcurr - t0)/duration
    return f(x)

def smooth_down(t0,tcurr,duration,t=0.2):
    def f(x): 
        return 1/exp(t*x)
    # 1 when tcurr = t0, 0 when starting
    x = (tcurr - t0)/duration
    print f(x)
    return f(x)


# Update stimulus input
@network_operation(current_clock, when='start')
def update_currents(current_clock):
    c_time = current_clock.t
    print c_time
    if time_cond(c_time,tc_start,tc_stop):
        Pe.i_e = current_e
    elif time_cond(c_time,reset_start,reset_stop):
       Pe.i_e = -4.5*bkg
       #print "w"
       #Pe.i_e = -40*bkg2*smooth_down(reset_start,c_time,reset_stop -reset_start)
    elif time_cond(c_time,restart_start,restart_stop):
        if drive:
            Pe.i_e = bkg2*smooth(restart_start,c_time,restart_stop -restart_start)
    elif time_cond(c_time,sec_cue_start, sec_cue_stop):
        Pe.i_e = current_e2
    else:
        Pe.i_e = 0

if drive:
    logfile = "read_out_log_bkg_%.2f.txt" % (bkg_fact)
else:
    logfile = "read_out_log_no_bkg.txt" 
 
if drive:
    log_file = "bkg_%.2f_%s" % (bkg_fact,log_file)
else:
    log_file = "no_bkg_%s" % log_file 
 
# Monitor spikes in the network:

spikes = FileSpikeMonitor(Pe,log_file+"_spikes.dat",record=True)
#spikes_i = SpikeMonitor(Pi,record=True)
#spikes = SpikeMonitor(Pe,record=True)

u_m = StateMonitor(Pe,'synu',timestep=1000,clock=simulation_clock,record=True)
x_m = StateMonitor(Pe,'synx',timestep=1000,clock=simulation_clock,record=True)

# Run the simulation:
run(simtime)

savetxt(log_file+"_u_m.dat",u_m.values)
savetxt(log_file+"_x_m.dat",u_m.values)
i,t = spikes.it
time=arange(max(t)+1)



fr1 = zeros(NE)
for n in range(NE):
    fr1[n] = 2.*sum(i[t>simtime-0.5*second] == n)


def decode(RE):
    N=NE
    R = []
    angles = np.arange(0,N)*2*np.pi/N
    R=sum(np.dot(RE,np.exp(1j*angles)))/n
    angle = np.angle(R)
    if angle < 0:
        angle +=2*pi 
    return angle 



w1=100*ms
w2=250*ms
n_wins = int((simtime-w2)/w1)
fr2 = zeros([NE,n_wins])

w_ang = radians(18)


N_i = sec_cue_neuron - ang2ne(w_ang)
N_f = sec_cue_neuron + ang2ne(w_ang)


Ns = []

decs=[]
for ti in range(int(n_wins)):
    if N_f - N_i < -ang2ne(w_ang)*3:
        N_f, N_i = N_i,N_f
    i_i = (i>N_i) & (i<N_f)
    if N_f - N_i > ang2ne(w_ang)*3:
        i_i = (i>N_f) | (i<N_i)     
    t_beg = ti*w1
    t_end = t_beg + w2
    t_i = (t>t_beg) & (t<t_end)
    i_2=i[t_i & i_i]
    t_2=t[t_i & i_i]
    fr = zeros(NE)
    for n in range(NE):
        fr[n] = sum(i_2 == n)
    dec=decode(fr)
    decs.append(dec)
    N_ix = ang2ne(circdist(dec,w_ang))[0]
    N_fx = ang2ne(circdist(dec,-w_ang))[0]
    Ns.append([N_ix,N_fx])
    if t_beg <sec_cue_start:
        continue
    N_i = N_ix
    N_f = N_fx

savetxt(log_file+"_decs.dat",decs)

   
dec = decs[-1]
#dec=decode(fr1)

with open(logfile, "a") as myfile:
    #myfile.write(str(radians(i_cue_ang))+" " +str(decode(fr1))+"\n")
    myfile.write(log_file+" "+str(radians(i_cue_ang))+" "+str(radians(sec_cue_ang))+" "+str(dec)+"\n")



plot(t,i,".",ms=2)
fill_between([reset_start,reset_stop],[0,0],[NE,NE],alpha=0.2,color="red")
fill_between([restart_start,restart_stop],[0,0],[NE,NE],alpha=0.2,color="green")
fill_between([sec_cue_start,sec_cue_stop],[0,0],[NE,NE],alpha=0.2,color="green")
fill_between([tc_start,tc_start+tc_dur],[0,0],[NE,NE],alpha=0.2,color="green")
plot([0,simtime/second],[sec_cue_neuron,sec_cue_neuron],"r--")
plot([0,simtime/second],[i_cue_ang/360.*NE,i_cue_ang/360.*NE],"k--")
plot(linspace(0,simtime/second,n_wins),array(decs)/(2*pi)*NE)
plot(linspace(0,simtime/second,n_wins),array(decs)/(2*pi)*NE,"r-",ms=10)

plt.show()