# This file generates the plots for the Simple Intervention
# Figures 6 and 7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from functions import adSEIR_pairwise, SEIR_pairwise, AIAT, inflection, local_max, event_times

from scipy.integrate import odeint#, solve_ivp


## use LaTeX fonts in the plot
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

### Bipartite generator with two Poisson Distributions
N = 10000
M = 2500
lamb1 = 4

#Faster Intials for Large Bipartite Projections
k_0 = (N/M)*(lamb1)**2
k2_k_0 = ((N/M)**2)*(lamb1**3)*(lamb1+1)
phi = 1/(lamb1+1)


# # Computing Random Variates from the Double Poisson
# # From Newman, Watts, Strogatz - Stirling's Approx is used for large factorials

# #Initialize a probability vector with p_0 = 0, which will be fixed to sum to 1 (we aren't worried about 0 degree nodes anyway)
# probs = [0]

# lamb2 = N*lamb1/M

# for k in range(1,N+1):
#    #degrees greater than 230 occur with negligible probability, and this avoids numerical issues
#     if k > 230:
#         probs.append(0)
#     else:
#         const = np.exp(lamb1*(np.exp(-lamb2)-1))/(2*np.pi*np.sqrt(k))
#         S = sum([const*((lamb2*np.exp(1)/k)**k)*((lamb1*np.exp(-lamb2+1))**i)*(i**(k-i-0.5)) for i in range(1,k)])
#         probs.append(S)

# probs = probs/sum(probs)

###############
# %% Figure 6 - Single Infection Curve

#Defining Time
years = 2 #years 
t_final = 365*years #number of days
t_steps = t_final #use timestep of one day
t = np.linspace(0, t_final, t_steps+1) #time vector 

#Epidemiological Parameters
eta = 1/5 #Rate of infection onset
gamma = 1/10 #Rate of Recovery
R_0 = 2.4 #Basic Reproductive Number
K = (k2_k_0)/k_0
beta = (-(2*R_0-K)*gamma - gamma*np.sqrt(K**2-4*R_0*K*phi) )/(2*(R_0-K+K*phi)) #Solved from Miller (2009)

#Initial Conditions - Randomly Select two nodes and get their degrees (one for E_0, one for I_0)
# Currently, these values are fixed for consistency
rand1 = 60 #rand1 = np.random.choice(range(0,N+1),p=probs)
rand2 = 48 #rand2 = np.random.choice(range(0,N+1),p=probs)

#Static Network Model
u_0 = [N-2,1,1,k_0*N-2*(rand1+rand2),rand1,rand2,0,0,0] #[S,E,I,SS,SE,SI,EE,EI,II]
sol = odeint(SEIR_pairwise,u_0,t,args=(beta,eta,gamma,N,k_0,k2_k_0,phi))
S, E, I = [sol[:,i] for i in [0,1,2]]
R = [N-S[i]-E[i]-I[i] for i in range(len(t))]

#Network Dynamics
#Intervention Parameters
p = 0.5 #severity
q = 0.01 #threshold

int_start = min([t[i] for i in range(len(t)) if I[i] > q*N]) #Starting time for intervention
L_I = 15 #length of intervention
L_H = 15 #length of holding
L_R = 150 #length of relaxation

#Times of Phase change
t_1 = int_start
t_2 = t_1+L_I
t_3 = t_2+L_H
t_4 = t_3+L_R

#Values of omega* and alpha*
omega = -np.log(p)/L_I 
alpha = np.log((N-1-p*k_0)/(N-1-k_0))/L_R

#Activation/Deletion Rate Functions
#Rate of link activation alpha(t) 
def a(t):
    if t_3 <= t < t_4:
        return(alpha)
    else:
        return(0)

#Rate of link deletion omega(t)
def w(t):
    if t_1 <= t < t_2:
        return(omega)
    else:
        return(0)

#Initial Condition for Adaptive SEIR Model
ad_u_0 = [N-2,1,1,k_0*N-2*(rand1+rand2),rand1,rand2,0,0,0,k_0,k2_k_0,phi] #[S,E,I,SS,SE,SI,EE,EI,II,k,k^2-k,phi]

#Solving the Adaptive SEIR Model
ad_sol = odeint(adSEIR_pairwise,ad_u_0,t,args=(beta,eta,gamma,a,w,N))
adS, adE, adI = [np.ndarray.tolist(ad_sol[:,i]) for i in [0,1,2]]
adR = [N-adS[i]-adE[i]-adI[i] for i in range(len(t))]


# %% Plots for Fig 6 - each panel can be made by adjusting appropriate parameters

fig, ax = plt.subplots(1,1,figsize=(4,4))

#Adjust axis and tick parameters
ax.set_xlim(-15,1.5*365)
ax.tick_params(axis='x', labelsize='large')
ax.tick_params(axis='y', labelsize='large')
ax.set_xlabel(r'$t$',fontsize=16)
ax.set_ylabel(r'[$I$]',fontsize=16)

#Add in the times for phase changes
ax.plot([t_1,t_1],[0,max(I)],color="lightgray",linestyle="dashed")
ax.plot([t_2,t_2],[0,max(I)],color="lightgray",linestyle="dashed")
ax.plot([t_3,t_3],[0,max(I)],color="lightgray",linestyle="dashed")
ax.plot([t_4,t_4],[0,max(I)],color="lightgray",linestyle="dashed")

#Plot Static and Adaptive Models
ax.plot(t,I,c="orangered",linestyle="dotted",label="No Intervention")
ax.plot(t,adI,c="orangered",label="Intervention")

fig.tight_layout()

#fig.savefig("Figure6a.eps")

# %% Figure 7 - Simple Intervention for Various Parameter Combinations

#Defining Time
years = 12 #years 
t_final = 365*years #number of days
t_steps = t_final #timestep of one day
t = np.linspace(0, t_final, t_steps+1) #time vector 

# Defining Parameters
L_H_const = 15 #days holding
L_I = [2*n for n in range(1,91)] #days of lockdown intervention/or the length of holding
L_R = [2*n for n in range(1,91)] #days of until return to normal

#Proportion of contacts at full social distancing - p
props = [0.125,0.25,0.5]

#Threshold Function
def threshold(t,y,q,N):
    return(y[2]-N*q)

def wrapper(t,y,beta,eta,gamma,a,w,N):
    return(threshold(t,y,q,N))

#Preallocating for the RCFS, Inflection Points, Local Maxima, and AIAT
all_final_sizes = []
all_infl = []
all_maxes = []
all_AIAT = []

for p in props:
    final_sizes = []
    infl = []
    maxes = []
    AIATs = []
    # Loop over lengths of intervention, recovery periods
    for I in L_I:
        temp = []
        temp_infl = []
        temp_maxes = []
        temp_AIAT = []
        print(I)
        for R_ in L_R:
            #Compute omega* and alpha*
            omega = -np.log(p)/I
            alpha = np.log((N-1-p*k_0)/(N-1-k_0))/R_
            #Define rate functions
            def a(t):
                if int_start + I + L_H_const <= t < int_start + I + L_H_const + R_:
                    return(alpha)
                else:
                    return(0)
            def w(t):
                if int_start <= t < int_start + I:
                    return(omega)
                else:
                    return(0)
            #Solve ODE
            ad_sol = odeint(adSEIR_pairwise,ad_u_0,t,args=(beta,eta,gamma,a,w,N))
            event_t = event_times(t,ad_sol,q,N) #get times of threshold cross
            #Compute RCFS
            R_temp = [N-ad_sol[i,0]-ad_sol[i,1]-ad_sol[i,2] for i in range(len(t))]
            temp.append((R_temp[-1]-R[-1])/R[-1])
            #Compute other metrics
            temp_maxes.append(local_max(t,ad_sol,beta,eta,gamma))
            temp_infl.append(inflection(t,ad_sol,beta,eta,gamma))
            temp_AIAT.append(AIAT(R_temp,event_t,t,gamma,q,N))
        final_sizes.append(temp)
        infl.append(temp_infl)
        maxes.append(temp_maxes)
        AIATs.append(temp_AIAT)
    all_final_sizes.append(final_sizes)
    all_infl.append(infl)
    all_maxes.append(maxes)
    all_AIAT.append(AIATs)
    
# %% Plot for Figure 7
    
R_period, I_period = np.meshgrid(L_R,L_I)

v_min=-0.4 #min([min([min(x) for x in z]) for z in all_final_sizes])
v_max=max([max([max(x) for x in z]) for z in all_final_sizes])
    
v_min_AIAT=min([min([min(x) for x in z]) for z in all_AIAT])
v_max_AIAT=max([max([max(x) for x in z]) for z in all_AIAT])
      
# # Number of Inflection Points
# v_min_waves = min([min([min(x) for x in z]) for z in all_infl])
# v_max_waves = max([max([max(x) for x in z]) for z in all_infl])

#Colormaps
cm = plt.cm.ScalarMappable(cmap = plt.get_cmap('viridis'), norm = mpl.colors.Normalize(vmin=v_min,vmax=v_max))
cm_AIAT = plt.cm.ScalarMappable(cmap = plt.get_cmap('plasma'), norm = mpl.colors.Normalize(vmin=v_min_AIAT,vmax=v_max_AIAT))

#Contour Levels
levels = list(np.linspace(v_min,v_max,20))
levels_AIAT = list(np.linspace(v_min_AIAT,v_max_AIAT,20))

fig, axs = plt.subplots(1,len(props),figsize=(len(props)*12,7))
fig_AIAT, axs_AIAT = plt.subplots(1,len(props),figsize=(len(props)*12,7))

for i in range(len(props)):
    # RCFS
    p = axs[i].contourf(I_period,R_period,all_final_sizes[i],cmap='viridis',levels=levels)
    axs[i].set_xlabel(r'$L_I$',fontsize=26)
    axs[i].set_ylabel(r'$L_R$',fontsize=26)
    axs[i].set_title(r'$p = $'+ str(props[i]),fontsize=30)
    axs[i].tick_params(axis='x', labelsize=25.0)
    axs[i].tick_params(axis='y', labelsize=25.0)

    # AIAT
    p = axs_AIAT[i].contourf(I_period,R_period,all_AIAT[i],cmap='plasma',levels=levels_AIAT)
    axs_AIAT[i].set_xlabel(r'$L_I$',fontsize=26)
    axs_AIAT[i].set_ylabel(r'$L_R$',fontsize=26)
    axs_AIAT[i].set_title(r'$p = $'+ str(props[i]),fontsize=30)
    axs_AIAT[i].tick_params(axis='x', labelsize=25.0)
    axs_AIAT[i].tick_params(axis='y', labelsize=25.0)
    
    #Qualitative Boundaries for p=0.25
    if i == 1:
        #In this case, the boundary for uniform vs. multiple spikes is most appropriate 
        p_infl = axs[i].contour(I_period,R_period,all_maxes[i],levels=[3,5],colors="white",linewidths=3.0)
    
    #Qualitative Boundaries for p=0.5
    elif i == 2:
        p_max = axs[i].contour(I_period,R_period,all_maxes[i],levels=[0.5,1.5],colors="white",linestyles="dashed",linewidths=3.0)
        #Uniform-Nonuniform Boundary
        #Get the contours without plotting - will have to eliminate one contour that is numerical artifact
        p_infl = plt.contour(I_period,R_period,all_infl[i],levels=[3,5],colors="white",linewidths=3.0)
        plt.close()
        qual_bound_path = p_infl.collections[0].get_paths()[0]
        qual_bound_vert = qual_bound_path.vertices
        #Have to transpose vertices
        axs[i].plot(np.transpose(qual_bound_vert)[0],np.transpose(qual_bound_vert)[1],color="white",linewidth=3.0)
    else:
        continue
    
#Colorbars    
cbar = fig.colorbar(cm, ax=axs.ravel().tolist())
cbar_AIAT = fig_AIAT.colorbar(cm_AIAT, ax=axs_AIAT.ravel().tolist())
cbar.ax.tick_params(labelsize=25.0)
cbar_AIAT.ax.tick_params(labelsize=25.0)
    
#fig.savefig("Figure7a.eps")
#fig_AIAT.savefig("Figure7b.eps")

