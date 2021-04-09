## This file generates a sample bipartite mixing network and corresponding
## unipartite contact network used for Figure 3

import networkx as nx
import matplotlib.pyplot as plt

from functions import bipartite_poisson_generator


# %% Generate bipartite network with two Poisson degree distributions

N = 200 #Number of Individuals
M = 50 #Number of Mixing Locations
lamb1 = 2 #Average Individual Degree (Bipartite Network)

G, G_bipartite = bipartite_poisson_generator(N,M,lamb1)

#Degree Distribution
degree_dist = list(nx.degree_histogram(G)).copy()
degrees = [d for d in range(len(degree_dist)) for i in range(degree_dist[d])]

#####################
# %% Plots


# Figure 3a - Mixing Network

plt.figure(figsize=(12,4))
color = ['#47aee6' for i in range(N)] + ['#123AA1' for i in range(M)]
size = [20 for i in range(N)] + [50 for i in range(M)]
nx.draw(G_bipartite,node_color=color, node_size = size, width=0.2,pos=nx.bipartite_layout(G_bipartite,nx.bipartite.sets(G_bipartite)[1],align='horizontal'))

#plt.savefig("Figure3a.png")

# Figure 3b - Contact Network

plt.figure(figsize=(10,10))
nx.draw(G, node_color='#47aee6', node_size = 40, width=0.2, with_labels=False, pos=nx.spring_layout(G))

#plt.savefig("Figure3b.png")

# Figure 3c - Degree Distribution Histogram

plt.figure(figsize=(10,10))
plt.hist(degrees, bins=[0.5+i for i in range(max(degrees)+1)],density=True)
plt.xticks(fontsize=20.0)
plt.yticks(fontsize=20.0)

#plt.savefig("Figure3c.png")

