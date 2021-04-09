# This file generates the plots for the Prevalence-Dependent Intervention
# Figures 10 - 14

#Necessary Packages

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from functions import SEIR_pairwise, PD_threshold_response, AIAT

## use LaTeX fonts in the plot
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# %% Contact Network and Epidemiological Parameters

### Bipartite generator with two Poisson Distributions
N = 10000
M = 2500
lamb1 = 4

## This code can be used to compute the probability distribution of degrees

## Computing Random Variates from the Double Poisson
## From Newman, Watts, Strogatz - Stirling's Approx is used for large factorials

# Initialize a probability vector with p_0 = 0, which will be fixed to sum to 1 (we aren't worried about 0 degree nodes anyway)

probs = [0]
lamb2 = N*lamb1/M
for k in range(1,N+1):
   #degrees greater than 230 occur with negligible probability, and this avoids numerical issues
    if k > 230:
        probs.append(0)
    else:
        const = np.exp(lamb1*(np.exp(-lamb2)-1))/(2*np.pi*np.sqrt(k))
        S = sum([const*((lamb2*np.exp(1)/k)**k)*((lamb1*np.exp(-lamb2+1))**i)*(i**(k-i-0.5)) for i in range(1,k)])
        probs.append(S)
probs = probs/sum(probs)

# From Newman, Watts, Strogatz and Me, we can compute the three important network parameters
k_0 = (N/M)*(lamb1)**2
k2_k_0 = ((N/M)**2)*(lamb1**3)*(lamb1+1)
phi = 1/(lamb1+1)

#Epidemiological Parameters
eta = 1/5 #Rate of infection onset
gamma = 1/10 #Rate of Recovery
R_0 = 2.4 #Basic Reproductive Number
K = (k2_k_0)/k_0
beta = (-(2*R_0-K)*gamma - gamma*np.sqrt(K**2-4*R_0*K*phi) )/(2*(R_0-K+K*phi))

#########################
# %% SINGLE INFECTION CURVE - Figures 10 and 12

#Intervention Parameters
p = 0.05
q = 0.002
L_I = 30
L_H = 0
L_R = 90

years = 25
t_final = years*365 

#Values of omega* and alpha*
omega = -np.log(p)/L_I 
alpha = np.log((N-1-p*k_0)/(N-1-k_0))/L_R

#Initial Infectioned and Exposed (need the degrees of two random nodes)
rand1 = np.random.choice(range(0,N+1),p=probs) 
rand2 = np.random.choice(range(0,N+1),p=probs) 

#Initial Condition
ad_u_0_s = [N-2,1,1,k_0*N-2*(rand1+rand2),rand1,rand2,0,0,0,k_0,k2_k_0,phi] #[S,E,I,SS,SE,SI,EE,EI,II,k,k^2,phi]

#Solve
times, solution, event_times = PD_threshold_response(ad_u_0_s,t_final,p,q,L_I,L_R,alpha,omega,beta,eta,gamma,N)
times = np.ndarray.tolist(times)

adS_s, adE_s, adI_s = [solution[i,:] for i in [0,1,2]]
adR_s = [N-adS_s[i]-adE_s[i]-adI_s[i] for i in range(len(times))]

# Static
u_0_s = [N-2,1,1,k_0*N-2*(rand1+rand2),rand1,rand2,0,0,0] #[S,E,I,SS,SE,SI,EE,EI,II]
sol_s = odeint(SEIR_pairwise,u_0_s,times,args=(beta,eta,gamma,N,k_0,k2_k_0,phi))
S_s, E_s, I_s = [sol_s[:,i] for i in [0,1,2]]
R_s = [N-S_s[i]-E_s[i]-I_s[i] for i in range(len(times))]

#Plotting the Curve

t_end = years*365 #Easily adjust the ending time for all the plots without having to mess with the blankspace around the plot
end = next(i for i in range(len(times)) if times[i] >= t_end)

fig, ax = plt.subplots(1,1,figsize=(4,4)) 

## PLOT I(t)
for t in event_times[:-1]:
    if t < t_end:
        ax.plot([t,t],[0,max(I_s)],color="lightgray",linestyle="dashed")
ax.plot([0,t_end],[q*N,q*N],color="lightgray",linestyle="dashed")
ax.plot(times[:end],I_s[:end],c="orangered",linestyle="dotted",label="I (Static)")
ax.plot(times[:end],adI_s[:end],c="orangered",label="I (Adaptive)")

ax.tick_params(axis='x', labelsize='large')
ax.tick_params(axis='y', labelsize='large')
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'$[I]$',fontsize=16)

fig.tight_layout()

#fig.savefig("Figure10c.eps")
#fig.savefig("Figure12c.eps")

################################
# %% Plots for Figures 11 and 13
    
# Defining Parameters
L_I =[2*n for n in range(1,91)] #days of lockdown intervention
L_R = [2*n for n in range(1,91)] #days of until return to normal
q_props = [0.005,0.01,0.02]
props = [0.125,0.25,0.5]

#Final Time
years = 12
t_final = years*365

all_final_sizes = []
all_AIAT = []

for q in q_props:
    for p in props:
        final_sizes = []
        AIATs = []
        # Loop over intervention, recovery
        for I in L_I:
            temp_final_sizes = []
            temp_AIAT = []
            for R in L_R:
                omega = -np.log(p)/I
                alpha = np.log((N-1-p*k_0)/(N-1-k_0))/R
                #
                times, ad_sol_s, event_times, distance_times = PD_threshold_response(ad_u_0_s,t_final,p,q,I,R,alpha,omega,beta,eta,gamma,N)
                times = np.ndarray.tolist(times)
                R_temp = [N-ad_sol_s[0,i]-ad_sol_s[1,i]-ad_sol_s[2,i] for i in range(len(times))]
                temp_final_sizes.append((R_temp[-1]-R_s[-1])/R_s[-1])
                temp_AIAT.append(AIAT(R_temp,event_times,times,gamma,q,N))
            final_sizes.append(temp_final_sizes)
            AIATs.append(temp_AIAT)
        all_final_sizes.append(final_sizes)
        all_AIAT.append(AIATs)

#######################
# Actual Plot        

R_period, I_period = np.meshgrid(L_R,L_I)

v_min= -0.4 #min([min([min(x) for x in z]) for z in all_final_sizes]) #-0.4
v_max=max([max([max(x) for x in z]) for z in all_final_sizes])

v_min_AIAT=min([min([min(x) for x in z]) for z in all_AIAT])
v_max_AIAT=max([max([max(x) for x in z]) for z in all_AIAT])


cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('viridis'), norm = mpl.colors.Normalize(vmin=v_min,vmax=v_max))
cm_AIAT = plt.cm.ScalarMappable(cmap = plt.get_cmap('magma'), norm = mpl.colors.Normalize(vmin=v_min_AIAT,vmax=v_max_AIAT))

levels = list(np.linspace(v_min,v_max,20))
levels_AIAT = list(np.linspace(v_min_AIAT,v_max_AIAT,20))


fig, axs = plt.subplots(len(q_props),len(props),figsize=(len(props)*15,len(q_props)*12))
fig_AIAT, axs_AIAT = plt.subplots(len(q_props),len(props),figsize=(len(props)*15,len(q_props)*12))

for j in range(len(q_props)):
    for i in range(len(props)):
        axs[2-j,i].contourf(I_period,R_period,all_final_sizes[len(q_props)*j+i], vmin = v_min, vmax=v_max, levels=levels)
        axs[2-j,i].tick_params(axis='x', labelsize=30.0)
        axs[2-j,i].tick_params(axis='y', labelsize=30.0)
        axs[2-j,i].set_xlabel(r'$L_I$',fontsize=30)
        axs[2-j,i].set_ylabel(r'$L_R$',fontsize=30)
        axs[2-j,i].set_title(r'$q = $'+ str(q_props[i])+r', $p = $'+ str(props[j]),fontsize=35)
        
        axs_AIAT[2-j,i].contourf(I_period,R_period,all_AIAT[len(q_props)*j+i],vmin = v_min_AIAT, vmax=v_max_AIAT,cmap='magma', levels=levels_AIAT)
        axs_AIAT[2-j,i].tick_params(axis='x', labelsize=30.0)
        axs_AIAT[2-j,i].tick_params(axis='y', labelsize=30.0)
        axs_AIAT[2-j,i].set_xlabel(r'$L_I$',fontsize=30)
        axs_AIAT[2-j,i].set_ylabel(r'$L_R$',fontsize=30)
        axs_AIAT[2-j,i].set_title(r'$q = $'+ str(q_props[i])+r', $p = $'+ str(props[j]),fontsize=35)
        
cbar = fig.colorbar(cm, ax=axs.ravel().tolist())
cbar_AIAT = fig_AIAT.colorbar(cm_AIAT, ax=axs_AIAT.ravel().tolist())

cbar.ax.tick_params(labelsize=30.0)
cbar_AIAT.ax.tick_params(labelsize=30.0)

#fig.savefig("Figure11.eps")
#fig_AIAT.savefig("Figure13.eps")

###############################
# %%  Plots for Figures 14 

# Defining Parameters
L_I = [15,30,60] 
L_R = 90 
q_props = [0.001*n for n in range(1,31)]
props = [0.025*n for n in range(1,41)]

#Final Time
years = 6 
t_final = years*365

all_final_sizes = []
all_AIAT = []

for I in L_I:
    final_sizes = []
    AIATs = []
    for p in props:
        temp_final_sizes = []
        temp_AIAT = []
        for q in q_props:
            omega = -np.log(p)/I
            alpha = np.log((N-1-p*k_0)/(N-1-k_0))/L_R
            times, ad_sol_s, event_times, distance_times = PD_threshold_response(ad_u_0_s,t_final,p,q,L_I,L_R,alpha,omega,beta,eta,gamma,N)
            times = np.ndarray.tolist(times)
            R_temp = [N-ad_sol_s[0,i]-ad_sol_s[1,i]-ad_sol_s[2,i] for i in range(len(times))]
            temp_final_sizes.append((R_temp[-1]-R_s[-1])/R_s[-1])
            temp_AIAT.append(AIAT(R_temp,event_times,times,gamma,q,N))
        final_sizes.append(temp_final_sizes)
        AIATs.append(temp_AIAT)
    all_final_sizes.append(final_sizes)
    all_AIAT.append(AIATs)
    
# Actual Plots

q_y, p_x = np.meshgrid(q_props,props)

v_min=min([min([min(x) for x in z]) for z in all_final_sizes])
v_max=max([max([max(x) for x in z]) for z in all_final_sizes])

v_min_AIAT=min([min([min(x) for x in z]) for z in all_AIAT])
v_max_AIAT=max([max([max(x) for x in z]) for z in all_AIAT])


cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('viridis'), norm = mpl.colors.Normalize(vmin=v_min,vmax=v_max))
cm_AIAT = plt.cm.ScalarMappable(cmap = plt.get_cmap('magma'), norm = mpl.colors.Normalize(vmin=v_min_AIAT,vmax=v_max_AIAT))

levels = list(np.linspace(v_min,v_max,20))
levels_AIAT = list(np.linspace(v_min_AIAT,v_max_AIAT,20))

fig, axs = plt.subplots(1,3,figsize=(30,7))
fig_AIAT, axs_AIAT = plt.subplots(1,3,figsize=(30,7))

for i in range(3):
    axs[i].contourf(p_x,q_y,all_final_sizes[i], vmin = v_min, vmax=v_max, levels=levels)
    axs[i].tick_params(axis='x', labelsize=20.0)
    axs[i].tick_params(axis='y', labelsize=15.0)
    axs[i].set_xlabel(r'$p$',fontsize=22)
    axs[i].set_ylabel(r'$q$',fontsize=22)
    axs[i].set_title(r"$L_I = $" + str(L_I[i]),fontsize=22)

    axs_AIAT[i].contourf(p_x,q_y,all_AIAT[i],vmin = v_min_AIAT, vmax=v_max_AIAT,cmap='magma', levels=levels_AIAT)
    axs_AIAT[i].tick_params(axis='x', labelsize=20.0)
    axs_AIAT[i].tick_params(axis='y', labelsize=15.0)
    axs_AIAT[i].set_xlabel(r'$p$',fontsize=22)
    axs_AIAT[i].set_ylabel(r'$q$',fontsize=22)
    axs_AIAT[i].set_title(r"$L_I = $" + str(L_I[i]),fontsize=22)

fig.colorbar(cm, ax=axs.ravel().tolist())
fig_AIAT.colorbar(cm_AIAT, ax=axs_AIAT.ravel().tolist())

# fig.savefig("Figure14a.eps")
# fig.savefig("Figure14b.eps")


