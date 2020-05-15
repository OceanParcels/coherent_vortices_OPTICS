# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:37:37 2020

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from particle_and_network_classes import trajectory_data,  bipartite_network, undirected_network, domain
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from mpl_toolkits.basemap import Basemap
import matplotlib
import matplotlib.colors
from sklearn.cluster import cluster_optics_dbscan, cluster_optics_xi

#parameters
coherence_time = 20 #in 5 day intervals
K_vectors=50 
K_embedding=50
min_samples=100
xi = 0.047
d_deg = 1

initial_domain_edges = (-30., 20, -40., -20.)
drifter_data = trajectory_data.from_npz("Agulhas_particles.npz", set_nans = False, 
                                         domain_type = "agulhas_domain", time_interval=5,
                                         initial_domain_edges = initial_domain_edges)
T = len(drifter_data.drifter_longitudes[0])

#construct matrix M
bin_spacing = [d_deg, d_deg]        
drifter_data.compute_symbolic_sequences(bin_size = bin_spacing, dt=5)

#define M(tau)
C = []
for i in range(coherence_time): C.append(drifter_data.compute_C(i))
M = C[0].dot(C[0].transpose())
for i in range(1, coherence_time): M += C[i].dot(C[i].transpose())
M = M.tocsr()

print("Share of non-zero elements: ",len(M.data) / (M.shape[0] * M.shape[1]))
plt.hist(M.data)
plt.show()

A = undirected_network(M)

w, u =  A.compute_spectrum_L_randomwalk(K=K_vectors, plot=True)
U = u[:,:K_embedding]
U = np.array([U[i]/np.sqrt(U[i].dot(U[i])) for i in range(len(U))])

"""
Figure with embedding vector components
"""

panel_labels = ['(b)','(c)','(d)', '(e)','(f)']

#Figure with eigenvectors
f = plt.figure(constrained_layout=True, figsize = (10, 4))
gs = f.add_gridspec(2, 3)
ax = f.add_subplot(gs[0, 0])
ax.plot(range(K_vectors), w,'o', c = 'darkslategrey', markersize=3)
ax.set_title(r'(a) eigenvalues of $L_r$')
plt.grid(True)

gs_s = [[0,1],[0,2],[1,0],[1,1],[1,2]]
u_indices = [1,10,20,30,40]


for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])

    drifter_data.scatter_position_with_labels_geo(ax, U[:,u_indices[i]], cbar=True, size=0.5, 
                                                    t=0, cmap = 'cividis')
    
    ax.set_title(panel_labels[i] + ' i = ' + str(u_indices[i]))

f.savefig("./figures/agulhas_eigenvectors"+ "_k" + str(K_embedding) + "d_deg_" + str(d_deg) + "coherence_time_" + str(coherence_time), dpi=300)


"""
K-means
"""
labels_kmeans = KMeans(n_clusters=K_clusters, random_state=0).fit(u[:,:K_embedding]).labels_
bounds = np.arange(-1.5,np.max(labels_kmeans)+1.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))

f = plt.figure(constrained_layout=True, figsize = (10, 2))
gs = f.add_gridspec(1, 3)
gs_s = [[0,0], [0,1],[0,2]]
t_indices = [0,10,20]
panel_labels = ['(a)','(b)','(c)']

for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])

    drifter_data.scatter_position_with_labels_geo(ax, labels_kmeans, cbar=False, size=0.5, 
                                                    t=t_indices[i], cmap = 'tab20', cbarticks = range(-1,np.max(labels_kmeans)+1),
                                                    norm=norm)
    
    ax.set_title(panel_labels[i] + ' t = ' + str(t_indices[i]), size=10)


f.savefig("./figures/agulhas_kmeans"+ "_k" + str(K_embedding) + "d_deg_" + str(d_deg) + "coherence_time_" + str(coherence_time), dpi=300)


"""
OPTICS
"""
optics_clustering = OPTICS(min_samples=min_samples, metric='euclidean').fit(U)
reachability = optics_clustering.reachability_
core_distances = optics_clustering.core_distances_
ordering = optics_clustering.ordering_
predecessor = optics_clustering.predecessor_


labels_xi, clusters_xi = cluster_optics_xi(reachability, predecessor, ordering, min_samples,
                                            xi=xi, predecessor_correction=True)

bounds = np.arange(-1.5,np.max(labels_kmeans)+1.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
panel_labels = ['(b)','(c)','(d)', '(e)','(f)']

#Optics results
f = plt.figure(constrained_layout=True, figsize = (8, 6))
gs = f.add_gridspec(3, 2)
ax = f.add_subplot(gs[0, 0])
ax.grid(True)
ax.scatter(range(len(reachability)), reachability[ordering], c=labels_xi[ordering], 
                cmap = 'tab20', s=.1, norm=norm)
ax.set_title(r'(a) reachability plot, $\xi = $' + str(xi), size=10)
ax.set_ylabel(r"$\epsilon$")
ax.set_xlabel("ordering")
ax.tick_params(labelsize=8)
gs_s = [[0,1],[1,0],[1,1], [2,0], [2,1]]
t_indices = [0,10,20,40,60]

for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])

    drifter_data.scatter_position_with_labels_geo(ax, labels_xi, cbar=False, size=0.1,
                                                  t=t_indices[i], cmap = 'tab20', cbarticks = range(-1,np.max(labels_xi)+1),
                                                  norm=norm)
    
    ax.set_title(panel_labels[i] + ' Day ' + str(int(t_indices[i]*5)), size=10)


f.savefig("./figures/agulhas_optics_clustering"+ "_k" + str(K_embedding) + "d_deg_" + str(d_deg) + "coherence_time_" + str(coherence_time), dpi=300)


"""
Color plot of reachabilities
"""
f = plt.figure(constrained_layout=True, figsize = (6, 3))
ax = f.add_subplot()
drifter_data.scatter_position_with_labels_geo(ax, reachability, cbar=True, size=2,
                                              t=0, cmap = 'cividis', cbarlabel='reachability distance')    
f.savefig("./figures/agulhas_reachability"+ "_k" + str(K_embedding) + "d_deg_" + str(d_deg) + "coherence_time_" + str(coherence_time), dpi=300)


"""
Animation
"""
drifter_data.animate_particles_geo(labels_xi, filename='figures/agulhas_animation'+ "_k" + str(K_embedding) + "d_deg_" + str(d_deg) + "coherence_time_" + str(coherence_time), 
                                    animation_time=100, norm=norm, cmap='tab20')

