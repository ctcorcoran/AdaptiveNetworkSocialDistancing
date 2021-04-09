import numpy as np
import networkx as nx
import random as rand
from scipy.integrate import solve_ivp

########################################
# Functions for Generating the Network #
########################################

def degree_sequence_fixer(sequence,mean):
    seq = sequence.copy() #avoid trouble
    zero_ind = [i for i in range(len(seq)) if seq[i] == 0] #get the indices of the zeros in the degree sequence
    for z in zero_ind:
        seq[z] = np.random.choice([x for x in seq if x != 0]) #draw from the nonzero degrees
    while sum(seq) != round(len(seq)*mean):
        #We'll use N*(avg deg) as the target value for the number of stubs (rounded just in case)
        rand_node = rand.choice(list(range(len(seq)))) #there has to  be a better way to do this...
        direction = np.sign(round(len(seq)*mean)-sum(seq)) #do we want to add a stub or remove a stub?
        if seq[rand_node] == 1 and direction == -1:
            continue #Don't disconnect a node
        else:
            seq[rand_node] = seq[rand_node] + direction 
    return(seq)

def bipartite_poisson_generator(N,M,lamb1):
    lamb2 = lamb1*N/M
    
    ind_seq = np.random.poisson(lamb1,N)
    mixing_seq = np.random.poisson(lamb2,M)

    degseq1 = degree_sequence_fixer(ind_seq,lamb1)
    degseq2 = degree_sequence_fixer(mixing_seq,lamb2)

    G_bipartite = nx.Graph(nx.bipartite.configuration_model(degseq1, degseq2)) #alternatively, an option with nx.Graph() that preserves exact degree distribution
    while nx.is_connected(G_bipartite) == False:
        G_bipartite = nx.Graph(nx.bipartite.configuration_model(degseq1, degseq2))
        
    G = nx.bipartite.projected_graph(G_bipartite,nx.bipartite.sets(G_bipartite)[0])
    return(G,G_bipartite)

############################
# Functions for Simulation #
############################

def adaptive_SEIR_simulation(G,beta,eta,gamma,a,w,init_exp,init_inf,t_max):
    
    N = nx.number_of_nodes(G)
    
    # Initialize activation/deletion rates
    alpha = a(0)
    omega = w(0)
    
    # Get edges/possible edges
    all_possible_edges = [(i,j) for i in nx.nodes(G) for j in nx.nodes(G) if i < j]
    edges = list(nx.edges(G))
    non_edges = list(set(all_possible_edges)-set(edges))
    
    # Initialize epidemic variables
    times,S,E,I,R = [[0],[N-len(init_exp+init_inf)],[len(init_exp)],[len(init_inf)],[0]]
    infected_nodes = init_inf.copy()
    exposed_nodes = init_exp.copy()
    recovered_nodes = []
    neighbors = []
    
    # Get all the neighbors of infected nodes
    for i in infected_nodes:
        temp = list(nx.neighbors(G,i))
        neigh_temp = [x for x in temp if x not in neighbors]
        neighbors = neighbors + neigh_temp 
    at_risk_nodes = [node for node in neighbors if node not in infected_nodes+exposed_nodes+recovered_nodes]
    
    # Get force of infection
    infectious_neighbors = [0 for i in range(N)]
    for i in at_risk_nodes:
        infectious_neighbors[i] = len(list(set(infected_nodes).intersection(set(list(nx.neighbors(G,i))))))
    infection_rate = [beta*i for i in infectious_neighbors]
    
    # Total Rates
    total_activation_rate = alpha*len(non_edges)
    total_deletion_rate = omega*len(edges)
    total_infection_rate = sum(infection_rate)
    total_latency_rate = eta*len(exposed_nodes)
    total_recovery_rate = gamma*len(infected_nodes)
    #
    total_rate = total_infection_rate+total_latency_rate+total_recovery_rate+total_activation_rate+total_deletion_rate
    time = np.random.exponential(1/total_rate) 

    
    while (time < t_max) and (total_rate > 0): 
        r = np.random.uniform(0,total_rate)
        #Recovery
        if r<total_recovery_rate:
            u = rand.choice(infected_nodes)
            infected_nodes = [inf for inf in infected_nodes if inf != u]
            recovered_nodes.append(u)
            for i in list(nx.neighbors(G,u)):
                if infectious_neighbors[i] == 1:
                    infectious_neighbors[i] = 0
                    at_risk_nodes.remove(i)
                elif infectious_neighbors[i] > 1:
                    infectious_neighbors[i] -= 1 
        #Become Infectious
        elif total_recovery_rate <= r < total_recovery_rate+total_latency_rate:
            u = rand.choice(exposed_nodes)
            exposed_nodes = [exp for exp in exposed_nodes if exp != u]
            infected_nodes.append(u)
            for i in list(nx.neighbors(G,u)):
                if i not in at_risk_nodes+infected_nodes+exposed_nodes+recovered_nodes:
                    infectious_neighbors[i] = 1
                    at_risk_nodes.append(i)
                elif i in at_risk_nodes:
                    infectious_neighbors[i] +=1
        #Exposure
        elif total_recovery_rate+total_latency_rate <= r < total_recovery_rate+total_latency_rate+total_infection_rate:
            probs = [beta*inf/total_infection_rate for inf in infectious_neighbors if inf != 0]
            u = np.random.choice(sorted(at_risk_nodes),p=probs)
            at_risk_nodes = [node for node in at_risk_nodes if node != u]
            exposed_nodes.append(u)
            infectious_neighbors[u] = 0
        #Link Activation
        elif total_recovery_rate+total_latency_rate+total_infection_rate <= r < total_recovery_rate+total_latency_rate+total_infection_rate+total_activation_rate:
            if len(edges) == N*(N-1)/2:
                pass
            else:
                #Get a random non-edge
                node1,node2 = rand.choice(non_edges)

                #Add the edge
                non_edges.remove((node1,node2))
                edges.append((node1,node2))
                
                if node1 in infected_nodes:
                    if node2 not in at_risk_nodes+infected_nodes+exposed_nodes+recovered_nodes:
                        at_risk_nodes.append(node2)
                        infectious_neighbors[node2] = 1
                    elif node2 in at_risk_nodes:
                        infectious_neighbors[node2] += 1
                elif node2 in infected_nodes:
                    if node1 not in at_risk_nodes+infected_nodes+exposed_nodes+recovered_nodes:
                        at_risk_nodes.append(node1)
                        infectious_neighbors[node1] = 1
                    elif node1 in at_risk_nodes:
                        infectious_neighbors[node1] += 1
        #Link Removal
        else:
            if len(edges) == 0:
                pass
            else:
                #Random Edge
                node1,node2 = rand.choice(edges)
                
                #Remove the edge
                non_edges.append((node1,node2))
                edges.remove((node1,node2))

                if node1 in at_risk_nodes and node2 in infected_nodes:
                    if infectious_neighbors[node1] == 1:
                        at_risk_nodes.remove(node1)
                        infectious_neighbors[node1] = 0
                    else:
                        infectious_neighbors[node1] -= 1
                elif node2 in at_risk_nodes and node1 in infected_nodes:
                    if infectious_neighbors[node2] == 1:
                        at_risk_nodes.remove(node2)
                        infectious_neighbors[node2] = 0
                    else:
                        infectious_neighbors[node2] -= 1
        #Update Variable Lists
        times.append(time)
        S.append(N-len(infected_nodes+exposed_nodes+recovered_nodes))
        E.append(len(exposed_nodes))
        I.append(len(infected_nodes))
        R.append(len(recovered_nodes))
        
        #Update Rates
        alpha = a(time)
        omega = w(time)
        total_activation_rate = alpha*len(non_edges)
        total_deletion_rate = omega*len(edges)
        total_latency_rate = eta*len(exposed_nodes)
        total_recovery_rate = gamma*len(infected_nodes)
        total_infection_rate = beta*sum(infectious_neighbors)
        total_rate = total_infection_rate+total_latency_rate + total_recovery_rate + total_activation_rate + total_deletion_rate
        
        if total_rate != 0:
            time = time + np.random.exponential(1/total_rate)
    return [times, S, E, I, R]

def ensemble(G,beta,eta,gamma,a,w,init_exp,init_inf,num_sim,t_final,t_steps,thresh,prune=True):
    time_lists,sim_results_S, sim_results_E, sim_results_I, sim_results_R = [[],[],[],[],[]]
    ep_count = 0 #successful epidemics
    run_count = 0 #Runs
    while ep_count < num_sim:
        times, S, E, I, R = adaptive_SEIR_simulation(G,beta,eta,gamma,a,w,init_exp,init_inf,t_final) #Simulate epidemic
        run_count += 1
        if max(I) <= thresh: #If we never reach 10 infections, or just go back down, don't record it - the code is written so I -> I+1 is the only one-step increase 
            continue
        else:
            ep_count = ep_count + 1 #Count the epidemic
            ind = I.index(thresh) #Get the list index of the first time the infecteds exceed the threshold
            times = [t - times[ind] for t in times[ind:]] #time now starts when the threshold is attained
            S,E,I,R = [S[ind:],E[ind:],I[ind:],R[ind:]]
            time_lists.append(times)
            sim_results_S.append(S)
            sim_results_E.append(E)
            sim_results_I.append(I)
            sim_results_R.append(R)
    
    # Averaging Simulations
    # This will require averaging in some standardized time window
    
    #If we want to prune all results to the length shortest simulation run
    if prune == True:
        prune_time = min([time[-1] for time in time_lists])
    else:
        prune_time = t_final
    times_std = np.linspace(0, t_final, t_steps+1) #same as the ODE
    delta_t = t_final/t_steps #size of step
    times_std = [t for t in times_std if t < prune_time+delta_t] #strictly less than ensures that if prune_time is a standard time, it is the last recorded time
    
    sim_mean_S, sim_mean_E, sim_mean_I, sim_mean_R = [[],[],[],[]] #initialize means storage
    
    for t in times_std: #loop over time
        temp_SEIR = [] #at time t, each sublist will be the SEIR... values from a simulation i.e. [[S_1,E_1,I_1,R_1,...],[S_2,E_2,I_2,R_2,...],...]
        for s in range(num_sim): #loop over simulations
            ind = time_lists[s].index(max(x for x in time_lists[s] if x<=t)) #this will get the index of event time just prior to the step in t_std
            temp_SEIR.append([sim_results_S[s][ind],sim_results_E[s][ind],sim_results_I[s][ind],sim_results_R[s][ind]])
        #Now we average over all simulation and add to sim_mean
        sim_mean_S.append(sum([temp_SEIR[i][0] for i in range(num_sim)])/num_sim)
        sim_mean_E.append(sum([temp_SEIR[i][1] for i in range(num_sim)])/num_sim)
        sim_mean_I.append(sum([temp_SEIR[i][2] for i in range(num_sim)])/num_sim)
        sim_mean_R.append(sum([temp_SEIR[i][3] for i in range(num_sim)])/num_sim)

    #For easier returning of information, the full results from the simulations will be a list of lists of lists
    # Level 1: time_vectors, S, E, I, R
    # Level 2: individual simulations
    # Level 3: time of individual simulation
    full_sim_results = [time_lists, sim_results_S, sim_results_E, sim_results_I, sim_results_R]
    
    return [times_std, sim_mean_S, sim_mean_E, sim_mean_I, sim_mean_R, full_sim_results, run_count] 

######################
# Functions for ODEs #
######################
    
def SEIR_pairwise(u,t,beta,eta,gamma,N,k,k2_k,phi):
    [S,E,I,SS,SE,SI,EE,EI,II] = u
    
    #Network Parameter in the triple closure
    K = (k2_k)/k**2
    
    #Triple Closure
    if I == 0:
        SSI = 0
        ESI = 0
        ISI = 0
    elif I != 0 and E == 0:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = 0
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))    
    else:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = K*(SE*SI/S)*(1-phi+phi*(N/k)*EI/(E*I))
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))
    
    #Differential Equations
    dS = -beta*SI
    dE = beta*SI-eta*E
    dI = eta*E-gamma*I
    dSS = -2*beta*SSI
    dSE = -eta*SE+beta*(SSI-ESI)
    dSI = eta*SE-gamma*SI-beta*SI-beta*ISI
    dEE = -2*eta*EE+2*beta*ESI
    dEI = eta*EE-(gamma+eta)*EI+beta*SI+beta*ESI
    dII = 2*eta*EI-2*gamma*II
    
    return([dS,dE,dI,dSS,dSE,dSI,dEE,dEI,dII])

def adSEIR_pairwise(u,t,beta,eta,gamma,a,w,N):
    [S,E,I,SS,SE,SI,EE,EI,II,k,k2_k,phi] = u
    
    #Time Dependent activation/deletion rates - must pass functions
    alpha = a(t)
    omega = w(t)

    #Network Parameter in the triple closure
    K = (k2_k)/k**2
    
    #Triple Closure
    if I == 0:
        SSI = 0
        ESI = 0
        ISI = 0
    elif I != 0 and E == 0:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = 0
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))    
    else:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = K*(SE*SI/S)*(1-phi+phi*(N/k)*EI/(E*I))
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))
    
    #Differential Equations
    dS = -beta*SI
    dE = beta*SI-eta*E
    dI = eta*E-gamma*I
    dSS = -2*beta*SSI +alpha*(S*(S-1)-SS)-omega*SS
    dSE = -eta*SE+beta*(SSI-ESI)+alpha*(S*E-SE)-omega*SE
    dSI = eta*SE-gamma*SI-beta*SI-beta*ISI+alpha*(S*I-SI)-omega*SI
    dEE = -2*eta*EE+2*beta*ESI+alpha*(E*(E-1)-EE)-omega*EE
    dEI = eta*EE-(gamma+eta)*EI+beta*SI+beta*ESI+alpha*(E*I-EI)-omega*EI
    dII = 2*eta*EI-2*gamma*II+alpha*(I*(I-1)-II)-omega*II
    dk = alpha*(N-1)-(alpha+omega)*k
    dk2_k = 2*alpha*(N-2)*k-2*(alpha+omega)*k2_k
    dphi = 3*alpha-(alpha+omega)*phi-2*alpha*(N-2)*(k/k2_k)*phi
    
    return([dS,dE,dI,dSS,dSE,dSI,dEE,dEI,dII,dk,dk2_k,dphi])

def adSEIR_pairwise_ivp(t,u,beta,eta,gamma,a,w,N):
    [S,E,I,SS,SE,SI,EE,EI,II,k,k2_k,phi] = u
    
    #Time Dependent activation/deletion rates - must pass functions
    alpha = a(t)
    omega = w(t)

    #Network Parameter in the triple closure
    K = (k2_k)/k**2
    
    #Triple Closure
    if I == 0:
        SSI = 0
        ESI = 0
        ISI = 0
    elif I != 0 and E == 0:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = 0
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))    
    else:
        SSI = K*(SS*SI/S)*(1-phi+phi*(N/k)*SI/(S*I))
        ESI = K*(SE*SI/S)*(1-phi+phi*(N/k)*EI/(E*I))
        ISI = K*((SI**2)/S)*(1-phi+phi*(N/k)*II/(I**2))
    
    #Differential Equations
    dS = -beta*SI
    dE = beta*SI-eta*E
    dI = eta*E-gamma*I
    dSS = -2*beta*SSI +alpha*(S*(S-1)-SS)-omega*SS
    dSE = -eta*SE+beta*(SSI-ESI)+alpha*(S*E-SE)-omega*SE
    dSI = eta*SE-gamma*SI-beta*SI-beta*ISI+alpha*(S*I-SI)-omega*SI
    dEE = -2*eta*EE+2*beta*ESI+alpha*(E*(E-1)-EE)-omega*EE
    dEI = eta*EE-(gamma+eta)*EI+beta*SI+beta*ESI+alpha*(E*I-EI)-omega*EI
    dII = 2*eta*EI-2*gamma*II+alpha*(I*(I-1)-II)-omega*II
    dk = alpha*(N-1)-(alpha+omega)*k
    dk2_k = 2*alpha*(N-2)*k-2*(alpha+omega)*k2_k
    dphi = 3*alpha-(alpha+omega)*phi-2*alpha*(N-2)*(k/k2_k)*phi
    
    return([dS,dE,dI,dSS,dSE,dSI,dEE,dEI,dII,dk,dk2_k,dphi])

########################
# Evaluation Functions #                                          
########################

#Inflection Point Counter (Single Curve):
def inflection(t,y,beta,eta,gamma):
    infl = 0
    for i in range(len(t)-1):
        if y[i,2] > 1:
            #Evaluate I''
            I_dotdot = beta*eta*y[i,5]-eta*(eta+gamma)*y[i,1]+(gamma**2)*y[i,2]
            I_dotdot_ = beta*eta*y[i+1,5]-eta*(eta+gamma)*y[i+1,1]+(gamma**2)*y[i+1,2]
            if (I_dotdot > 0 and I_dotdot_ < 0) or (I_dotdot < 0 and I_dotdot_ > 0):
                infl += 1
    return(infl)

#Local Max Counter
def local_max(t,y,beta,eta,gamma):
    loc_max = 0
    for i in range(len(t)-1):
        if y[i,2] > 1:
            #Evaluate I'
            I_dot = eta*y[i,1]-gamma*y[i,2]
            I_dot_ = eta*y[i+1,1]-gamma*y[i+1,2]
            if (I_dot > 0 and I_dot_ < 0):
                loc_max += 1
    return(float(loc_max))

#Threshold Crossing Times
def event_times(t,y,q,N):
    event_times = []
    for i in range(1,len(t)):
        if (y[i-1,2] < q*N and y[i,2] > q*N) or ((y[i-1,2] > q*N and y[i,2] < q*N)):
            event_times.append(t[i])
    return(event_times)


# Average Infections Above Threshold
def AIAT(R,event_times,times,gamma,q,N):
    CIAT = 0
    t_tot = 0
    for i in range(int(len(event_times)/2)):
        t_start = event_times[2*i]
        t_end = event_times[2*i+1]
        ind_start = next(i for i in range(len(times)) if times[i] >= t_start) #times.index(t_start)
        ind_end = next(i for i in range(len(times)) if times[i] >= t_end) #times.index(t_end)
        CIAT += ((R[ind_end] - R[ind_start])/gamma-q*N*(t_end-t_start))
        t_tot += t_end-t_start
    return(CIAT/t_tot)


###############################################
# Functions for Prevalence-Dependent Response #   
###############################################
    
def PD_threshold_response(init,t_final,p,q,L_I,L_R,alpha,omega,beta,eta,gamma,N):
    # Initial Activation/Deletion Rate Functions
    # Both are zero until the threshold is crossed
    
    # Rate of link activation alpha(t) 
    def a(t):
        return(0)

    #Rate of link deletion omega(t)
    def w(t):
        return(0)
    
    #Define the event functions that will stop solve_ivp
    #q_up and q_down help make sure an event isn't immediately triggered on restart
    q_up = q
    q_down = q
        
    #Threshold function - decreasing through
    def threshold_decrease(t,y,q_down,N):
        return(y[2]-N*q_down)

    def wrapper_decrease(t,y,beta,eta,gamma,a,w,N):
        return(threshold_decrease(t,y,q_down,N))

    wrapper_decrease.terminal = True
    wrapper_decrease.direction = -1
    
    #Threshold function - increasing through
    def threshold_increase(t,y,q_up,N):
        return(y[2]-N*q_up)

    def wrapper_increase(t,y,beta,eta,gamma,a,w,N):
        return(threshold_increase(t,y,q_up,N))

    wrapper_increase.terminal = True
    wrapper_increase.direction = 1
    
    #Terminate 
    def terminate(t,y,beta,eta,gamma,a,w,N):
        return(y[1]+y[2])
    
    terminate.terminal = True

    #Initialize arrays for times and solution
    times = np.empty((0,)) #are 1D arrays columnless?
    solution = np.empty((len(init),0))
    
    #Initialize event times - threshold crossing
    t_event = 0 
    event_times = []
    
    E, I = init[1,2]
        
    while E + I >= 1 and t_event < t_final: #Makes sure the epidemic can still spread when an event occurs
        #Start the solver until the threshold hits
        sol = solve_ivp(adSEIR_pairwise_ivp,[t_event,t_final],init,events=(wrapper_increase,wrapper_decrease,terminate),args=(beta,eta,gamma,a,w,N))
        t_ad_s = sol.t
        ad_sol_s = sol.y #state x time
        E = ad_sol_s[1,-1]
        I = ad_sol_s[2,-1]

        #Add results - concatenate ndarrays
        times = np.concatenate((times,t_ad_s))
        solution = np.concatenate((solution,ad_sol_s),axis=1)

        #Note the threshold time and state
        t_event = t_ad_s[-1].copy()
        
        #Initialize the next stage of the solution with the last state before the event
        init = ad_sol_s[:,-1].copy()

        #For the condition, we can use I' to indicate whether we are increasing or decreasing through the threshold:
        # I' = eta*E - gamma*I
        dI = eta*ad_sol_s[1,-1]-gamma*ad_sol_s[2,-1]
        
        #Increasing
        if dI > 0:
            #A distancing event occurs
            event_times.append(t_event)
            q_up = 0 #This ensures that the threshold_increase doesn't trigger next
            q_down = q
            #Start social distancing
            #We adjust the length of the intervention period so it decreases to p*k_0 at the same rate
            L_I_adj = (1/omega)*np.log(ad_sol_s[9,-1]/(p*solution[9,0]))
            #Redefine the rate functions
            def w(t):
                if t_event <= t < t_event+L_I_adj:
                    return(omega)
                else:
                    return(0)
                
            def a(t):
                return(0)
        #Decreasing
        elif dI < 0:
            event_times.append(t_event)
            q_up = q
            q_down = 0 #Now we don't want threshold_decrease to trigger
            
            #If the intervention hasn't finished, we let it finish, otherwise L_I_adj = 0
            L_I_adj = (1/omega)*np.log(ad_sol_s[9,-1]/(p*solution[9,0]))
            #Redefine rate functions
            def w(t):
                if t_event <= t < t_event+L_I_adj:
                    return(omega)
                else:
                    return(0)
            
            def a(t):
                if t_event + L_I_adj <= t < t_event+L_I_adj+L_R:
                    return(alpha)
                else:
                    return(0)
        else: #On the off-chance that dI==0, stop the function
            break
    return([times,solution,event_times])
