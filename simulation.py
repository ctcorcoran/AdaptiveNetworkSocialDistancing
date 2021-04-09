## This file simulates an SEIR epidemic on a network
## used to create Figure 2

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as rand
from functions import bipartite_poisson_generator, adSEIR_pairwise, ensemble

from scipy.integrate import odeint

############################## 
# %% Bipartite generator with two Poisson Distributions
N = 500
M = 125
lamb1 = 4

G, _ = bipartite_poisson_generator(N,M,lamb1)

degree_dist = list(nx.degree_histogram(G)).copy()

###################
# %% Model Parameters
    
# Defining Time
t_final = 200
t_steps = 100
t = np.linspace(0, t_final, t_steps+1) #time vector 

# Network Parameters
k_0 = sum([degree_dist[k]*k for k in range(len(degree_dist))])/N
k2_k_0 = sum([degree_dist[k]*k*(k-1) for k in range(len(degree_dist))])/N
phi = nx.transitivity(G)

# Epidemiological Parameters
eta = 1/5
gamma = 1/10
R_0 = 2.4
K = k2_k_0/k_0
beta = (-(2*R_0-K)*gamma - gamma*np.sqrt(K**2-4*R_0*K*phi) )/(2*(R_0-K+K*phi))

# Network Dynamics Parameters 
alpha = 50/(N*(k2_k_0+k_0))
omega = 1/(N*k_0)

# Note that the activation and deletion rates are passed as functions
def a(t):
    return(alpha)

def w(t):
    return(omega)

#Initial Conditions 
init = 10 #Initial number of exposed and infectious nodes
init_E = sorted(rand.sample(range(N),init)) #Randomly select exposed
init_I = sorted(rand.sample(set(range(N))-set(init_E),init)) #Randomly select infectious

#Total degree of exposed and infectious nodes
deg_E = sum([G.degree(exp) for exp in init_E])
deg_I = sum([G.degree(inf) for inf in init_I])

#Initialize edge types
edges = list(nx.edges(G))
init_EE = 2*len(set([(a,b) for a in init_E for b in init_E if b>a]).intersection(set(edges)))
init_EI = len(set([(min(a,b),max(a,b)) for a in init_E for b in init_I]).intersection(set(edges)))
init_II = 2*len(set([(a,b) for a in init_I for b in init_I if b>a]).intersection(set(edges)))
init_SE = deg_E-init_EE-init_EI
init_SI = deg_I-init_II-init_EI
init_SS = N*k_0-(init_EE+init_II+2*(init_SE+init_SI+init_EI))

#Initial Condition for ODE
adu_0_s = [N-2*init,init,init,init_SS,init_SE,init_SI,init_EE,init_EI,init_II,k_0,k2_k_0,phi] #[S,E,I,SS,SE,SI,EE,EI,II,k,k^2-k,phi]

#Solve ODE
ode_solution = odeint(adSEIR_pairwise,adu_0_s,t,args=(beta,eta,gamma,a,w,N))

t = np.ndarray.tolist(t) #Easier to deal with a list

adS, adE, adI = [ode_solution[:,0],ode_solution[:,1],ode_solution[:,2]]
adR = [N-adS[i]-adE[i]-adI[i] for i in range(len(t))]


# %% Simulation

# Simulation Parameters
num_sim = 100 #Number of simulations to run in the ensemble 
thresh = init #Threshold value for infections - all simulations are time-shifted so time starts when I(t) = thresh
prune = False #Whether or not we want to cut off all runs to the length of the shortest simulation or not

# Run the simulation ensemble
times_std, sim_mean_S, sim_mean_E, sim_mean_I, sim_mean_R,  full_sim, runs = ensemble(G,beta,eta,gamma,a,w,init_E,init_I,num_sim,t_final,t_steps,thresh,prune)

#Shifting Time
I_temp = np.ndarray.tolist(adI)
ind = I_temp.index(next(x for x in I_temp if x >= thresh))
t_ODE_shift = [s-t[ind] for s in t[ind:]]
t_ODE_shift = t_ODE_shift[:len(I_temp[ind:])]

# %% Plots for Figure 2

fig1, ax1 = plt.subplots(1,1,figsize=(10,10))
fig2, ax2 = plt.subplots(1,1,figsize=(10,10))


for sim in range(num_sim):
    ax1.plot(full_sim[0][sim],full_sim[3][sim], c="lightgray")
    ax2.plot(full_sim[0][sim],full_sim[4][sim], c="lightgray")


ax1.plot(times_std,sim_mean_I,color='black')
ax2.plot(times_std,sim_mean_R,color='black')


ax1.set_xlabel(r'$t$',fontsize=26)
ax1.set_ylabel(r'[$I$]',fontsize=26)
ax2.set_xlabel(r'$t$',fontsize=26)
ax2.set_ylabel(r'[$R$]',fontsize=26)

ax1.tick_params(axis='x', labelsize=20.0)
ax1.tick_params(axis='y', labelsize=20.0)
ax2.tick_params(axis='x', labelsize=20.0)
ax2.tick_params(axis='y', labelsize=20.0)

ax1.plot(t_ODE_shift,adI[ind:],color='orangered',marker="o")
ax2.plot(t_ODE_shift,adR[ind:],color='forestgreen',marker="o")

#fig1.savefig("Figure2a.eps")
#fig1.savefig("Figure2b.eps")