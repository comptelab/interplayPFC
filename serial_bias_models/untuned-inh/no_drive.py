
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

fig_size = zeros(2,)
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = [fig_size[0],fig_size[1]]

bkg_facts = arange(0.1,2.1,0.1)
reset_durs = arange(100,2100,100)

bkg_fact = bkg_facts[int(random.random()*(len(bkg_facts)-1))]
reset_dur = reset_durs[int(random.random()*(len(reset_durs)-1))]


log_file = "simulation"+"_"+str(os.getpid())
my_file = Path(log_file)

i=0
while my_file.is_file():
    i+=1
    my_file = Path(log_file+str(i))

log_file+=str(i)

# print log_file


# ang = float(sys.argv[1])
fact = 1.275#1.28

# Here are the parameters of the simulation

# In[2]:

# size of the network
NE          = 1024              # number of E cells 
NI          = 256               # number of I cells 

# simulation-related parameters  
dt          = 0.02*ms     	# simulation step length [ms]
simulation_clock=Clock(dt=dt)   # clock for the time steps


# cue parameters
tc_start      = 1000*ms           # start of the cue signal
tc_dur        = 500*ms            # length period of the cue signal
i_cue_amp     = 0.25*0.25*nA            # cue current amplitude


i_cue_ang     = 180         # mean angle of the cue current
i_cue_width   = 60#14.4              # sigma/width of the current signal in degrees
dt_curr       = 50*ms             # current_clock step length (should be the highest common denominator between tc_start and tr_stop)
t_delay	      = 1000*ms		# delay duration

bkg_fact = 1.9
reset_dur = 800

reset_start = tc_start+tc_dur+t_delay
reset_stop = reset_start+750*ms
restart_start=reset_stop+500*ms
restart_stop = restart_start+300*ms

bump_dur = 0.5*ms

sec_cue_start=reset_stop+bump_dur
sec_cue_stop = sec_cue_start+tc_dur
sec_cue_ang = uniform(0,360)

simtime     =sec_cue_stop+t_delay     # total simulation time [ms] 



# pyramidal cells
Cm_e        = 0.5*nF  		# [nF] total capacitance
gl_e        = 25.0*nS 		# [ns] total leak conductance
El_e        = -70.0*mV		# [mV] leak reversal potential
Vth_e       = -50.0*mV		# [mV] threshold potential
Vr_e	    = -60.0*mV		# [mV] resting potential
tr_e	    = 2.0*ms		# [ms] refractory time

# interneuron cells
Cm_i        = 0.2*nF		# [nF] total capacitance
gl_i        = 20.0*nS		# [ns] total leak conductance
El_i        = -70.0*mV 		# [mV] leak reversal potential
Vth_i       = -50.0*mV 		# [mV] threshold potential
Vr_i	    = -60.0*mV 		# [mV] resting potential
tr_i	    = 1.0*ms  		# [ms] refractory time

# external background input
fext_e	    = 1800.0*Hz*ones(NE)		# [Hz] external input frequency (poisson train)
fext_i	    = 1800.0*Hz*ones(NI)
fext=append(fext_e,fext_i)

# AMPA receptor (APMAR)
E_ampa 	    = 0.0*mV  		# [mV] synaptic reversial potential
t_ampa	    = 2.0*ms  		# [ms] exponential decay time constant 
g_ext_e	    = 0.95*3.1*nS#0.975*3.1*nS  		# [nS] maximum conductance from external to pyramidal cells
g_ext_i	    = 2.38*nS 		# [nS] maximum conductance from external to interneuron cells
Gee_ampa    = 0.251*nS  #0.251*nS #0.391*nS      # [nS] maximum conductance from pyramidal to pyramidal cells
Gei_ampa    = 0.192*nS  #0.192*nS #0.293*nS      # [nS] maximum conductance from pyramidal to inhbitory cells

# GABA receptor (GABAR)
E_gaba 	    = -70.0*mV	  	# [mV] synaptic reversial potential
t_gaba	    = 10.0*ms		# [ms] exponential decay time constant 
Gie	    = 1.336*nS          # [ns] synaptic conductance interneuron to pyramidal cells
Gii	    = 1.024*nS          # [ns] synaptic conductance interneuron to interneuron cells

# NMDA receptor (NMDAR)
E_nmda 	    = 0.0*mV	 	# [mV] synaptic reversial potential
t_nmda      = 100.0*ms		# [ms] decay time of NMDA currents
t_x	    = 2.0*ms		# [ms] controls the rise time of NMDAR channels
alpha       = 0.5*kHz	  	# [kHz] controls the saturation properties of NMDAR channels
a           = 0.062/mV          # [1/mV] control the voltage dependance of NMDAR channel
b           = 1.0/3.57          # [1] control the voltage dependance of NMDAR channel ([Mg2+]=1mM )
Gee	    = fact*0.28*nS #0.381*nS    #0.274*nS      # [ns] synaptic conductance from pyramidal to pyramidal cells
Gei	    = 0.212*nS #0.292*nS   #0.212*nS       # [ns] synaptic conductance from pyramidal to interneuron cells

# Connection parameters
Jp_ee 	    = 1.62	  	# maximum connection weight from pyramidal to pyramidal
sigma_ee    = 14.4	 	# width of the connectivity footprint from pyramidal to pyramidal


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

weight=lambda i:(Jm_ee+(Jp_ee-Jm_ee)*exp(-0.5*(360.*min(i,NE-i)/NE)**2/sigma_ee**2))

weight_e=zeros(NE)
for i in xrange(NE): 
    weight_e[i]=weight(i)

fweight = rfft(weight_e) # Fourier transform

del weight_e


# function for circular boundary conditions

# return the normalised "distance" of the neurons in a circle
def circ_distance(fi):
    if (fi > 0):
        return min(fi,360-fi)
    else:
        return max(fi,fi-360)

def circ_distance(i,j):
    return degrees(circdist(i/NE*2*pi,j/NE*2*pi))



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
'''

eqs_i = '''
dv/dt = (-gl_i*(v-El_i)-g_ext_i*s_ampa*(v-E_ampa)-Gei*(v-E_nmda)*s_tot/(1+b*exp(-a*v))-Gii*s_gaba*(v-E_gaba))/Cm_i: volt 
ds_ampa/dt = -s_ampa/t_ampa : 1
ds_gaba/dt = -s_gaba/t_gaba : 1
s_tot : 1
'''


# Setting up the populations:

Pe = NeuronGroup(NE, eqs_e, threshold=Vth_e, reset=Vr_e, refractory=tr_e, clock=simulation_clock, order=2, freeze=True)
Pe.v = El_e
Pe.s_ext = 0
Pe.s_gaba = 0
Pe.s_nmda = 0
Pe.x = 0
Pe.i_e = 0

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

Cie = Connection(Pi, Pe, 's_gaba', weight=1.0)
Cii = Connection(Pi, Pi, 's_gaba', weight=1.0)

selfnmda=IdentityConnection(Pe, Pe, 'x', weight=1.0)

Cei_ampa = Connection(Pe,Pi,'s_ampa',weight=wrec_i)
Cee_ampa = Connection(Pe,Pe,'s_ampa',weight=lambda i,j:wrec_e*(Jm_ee+(Jp_ee-Jm_ee)*exp(-0.5*(360.*min(abs(i-j),NE-abs(i-j))/NE)**2/sigma_ee**2)))
stp=STP(Cee_ampa,taud=200*ms,tauf=1500*ms,U=0.2)
# set u to the st
stp.vars.u=0.43200000000000383
stp.vars.x=0.7486291487954766


# NMDA calculation to be performed at each time step:

# Calculate NMDA contributions
@network_operation(simulation_clock, when='start')
def update_nmda(simulation_clock):
    fsnmda = rfft(Pe.s_nmda) 
    fstot = fsnmda*fweight
    Pe.s_tot = irfft(fstot, NE)
    Pi.s_tot = fsnmda[0]

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


# Update stimulus input
@network_operation(current_clock, when='start')
def update_currents(current_clock):
    c_time = current_clock.t
    print c_time
    if time_cond(c_time,tc_start,tc_stop):
        Pe.i_e = current_e
    elif time_cond(c_time,reset_start,reset_stop):
        Pe.i_e = -bkg
    #elif time_cond(c_time,restart_start,restart_stop):
        #Pe.i_e = bkg2*(c_time-restart_start)/restart_start
#        Pe.i_e = bkg2*smooth(restart_start,c_time,restart_stop -restart_start)
    elif time_cond(c_time,sec_cue_start, sec_cue_stop):
        Pe.i_e = current_e2
    else:
        Pe.i_e = 0



# Monitor spikes in the network:

spikes = FileSpikeMonitor(Pe,log_file+"_spikes.dat",record=True)
#spikes = SpikeMonitor(Pe,record=True)
#u_m = StateMonitor(stp.vars,'u',timestep=1000,clock=simulation_clock,record=True)
#x_m = StateMonitor(stp.vars,'x',timestep=1000,clock=simulation_clock,record=True)



# Run the simulation:
run(simtime)


i,t = spikes.it

# w1=100*ms
# w2=500*ms
# n_wins = (simtime-w2)/w1
# fr2 = zeros([NE,n_wins])

# for ti in range(int(n_wins)):
#     t_beg = ti*w1
#     t_end = t_beg + w2
#     for n in range(NE):
#         fr2[n,ti] = sum(i[(t>=t_beg) & (t<t_end)] == n)

# n = int(i_cue_ang/360*NE)
# plot(linspace(0,6,len(u_m[n])),mean(u_m[n-10:n+10],0),label="syn facilitation");
# plot(linspace(0,6,len(fr2[n])),mean(fr2[n-10:n+10,:],0),label="firing rate")
# legend()

fr = zeros(NE)
for n in range(NE):
    fr[n] = sum(i[t>simtime-1*second] == n)


def decode(RE):
    N=1024
    R = []    
    angles = np.arange(0,N)*2*np.pi/N    
    R=sum(np.dot(RE,np.exp(1j*angles)))/n
    return (np.angle(R),np.abs(R))

#savetxt(log_file+"_u.dat",u_m.values)
#savetxt(log_file+"_x.dat",x_m.values)

with open("read_out_log.txt", "a") as myfile:
     myfile.write(str(radians(sec_cue_ang))+","+str(decode(fr)[0])+"\n")
