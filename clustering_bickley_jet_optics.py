"""
Bickley jet
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data,  undirected_network
from sklearn.cluster import KMeans, OPTICS
import matplotlib
import matplotlib.colors
from sklearn.cluster import  cluster_optics_xi

#parameters
r0 = 6371.
coherence_time = 401

#binning dimension (4 -> 16 particles per bin initially)
r=4
# K_clusters = 9 #number of eigenvectors used for embedding
K_embedding = 9
K_vectors = 30
min_samples=100
xi = 0.05

dx, dy = 100.57832561*r, 101.69491525*r
bin_spacing = [dx, dy]
drifter_trajectory_data = "bickley_jet_trajectories.npz"        
domain_edges = (0., np.pi * r0, -5000, 5000) # full domain, to cover all points along any trajectory        
drifter_data = trajectory_data.from_npz(drifter_trajectory_data, set_nans = False, 
                                        domain_type = "bickley_jet_domain", initial_domain_edges = domain_edges)
drifter_data.compute_symbolic_sequences(bin_size = bin_spacing)

#define M(tau)
C = drifter_data.compute_C(0)
M = C.dot(C.transpose())
for i in range(1, coherence_time): 
    C = drifter_data.compute_C(i)
    M += C.dot(C.transpose())

M = M.tocsr()
print("Share of non-zero elements: ",len(M.data) / (M.shape[0] * M.shape[1]))

A = undirected_network(M)    

w, u = A.compute_spectrum_L_randomwalk(K=K_vectors, plot=True) 
U = u[:,:K_embedding]
U = np.array([U[i]/np.sqrt(U[i].dot(U[i])) for i in range(len(U))]) #normalize


"""
Figure with embedding vector components
"""

f = plt.figure(constrained_layout=True, figsize = (10, 4))
gs = f.add_gridspec(2, 3)
panel_labels = ['(b)','(c)','(d)', '(e)','(f)']
ax = f.add_subplot(gs[0, 0])
ax.plot(range(K_vectors), w,'o', c = 'darkslategrey', markersize=3)
ax.set_title(r'(a) eigenvalues of $L_r$')
plt.grid(True)

gs_s = [[0,1],[0,2],[1,0],[1,1],[1,2]]
u_indices = [1,2,3,4,5]

for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])
    drifter_data.scatter_position_with_labels_flat(ax, U[:,u_indices[i]], colbar=True, size=0.5, 
                                                   t=0, cmap = 'cividis')
    ax.set_title(panel_labels[i] + ' i = ' + str(u_indices[i]))
      
    #axis ticks
    if i==0 or i==2: 
        plt.yticks(np.arange(-2000,4000,2000), np.arange(-2,4,2))
        ax.set_ylabel("y (1000 km)")
    else: plt.yticks([])
    
    if i>1: 
        plt.xticks(np.arange(0,25000,5000), np.arange(0,25,5))
        ax.set_xlabel("x (1000 km)")
    else: plt.xticks([])

f.savefig("./figures/bj_eigenvectors_r_" + str(r), dpi=300)


"""
K-means
"""
labels_kmeans = KMeans(n_clusters=K_embedding, random_state=0).fit(u[:,:K_embedding]).labels_
bounds = np.arange(-.5,np.max(labels_kmeans)+1.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))

f = plt.figure(constrained_layout=True, figsize = (10, 2))
gs = f.add_gridspec(1, 3)
panel_labels = ['(a)','(b)','(c)']
gs_s = [[0,0], [0,1],[0,2]]
t_indices = [0,50,200]

for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])

    drifter_data.scatter_position_with_labels_flat(ax, labels_kmeans, colbar=False, size=0.5, 
                                                   t=t_indices[i], cmap = 'tab20', cbarticks = range(-1,np.max(labels_kmeans)+1),
                                                   norm=norm)
    
    ax.set_title(panel_labels[i] + ' t = ' + str(t_indices[i]), size=10)
    ax.tick_params(labelsize=8)
      
    if i==0: 
        plt.yticks(np.arange(-2000,4000,2000), np.arange(-2,4,2))
        ax.set_ylabel("y (1000 km)")
    else: plt.yticks([])
    
    plt.xticks(np.arange(0,25000,5000), np.arange(0,25,5))
    ax.set_xlabel("x (1000 km)")
    
f.savefig("./figures/bj_kmeans_r_" + str(r) + "K_" + str(K_embedding), dpi=300)


"""
OPTICS
"""
optics_clustering = OPTICS(min_samples=min_samples, metric="euclidean").fit(U)

reachability = optics_clustering.reachability_
core_distances = optics_clustering.core_distances_
ordering = optics_clustering.ordering_
predecessor = optics_clustering.predecessor_

labels_xi, clusters_xi = cluster_optics_xi(reachability, predecessor, ordering, min_samples, xi=xi,
                      predecessor_correction=True)
bounds = np.arange(-1.5,np.max(labels_xi)+1.5,1)
norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))

#Optics results
f = plt.figure(constrained_layout=True, figsize = (7, 3))
gs = f.add_gridspec(2, 2)
ax = f.add_subplot(gs[0, 0])
panel_labels = ['(b)','(c)','(d)', '(e)','(f)']
ax.scatter(range(len(reachability)), reachability[ordering], c=labels_xi[ordering], 
                cmap = 'tab20', marker="o", s=.1, norm=norm)
ax.set_title(r'(a) reachability plot, $\xi = $' + str(xi), size=10)
ax.set_ylabel(r"$\epsilon$")
ax.set_xlabel("ordering")
ax.tick_params(labelsize=8)
gs_s = [[0,1],[1,0],[1,1]]
t_indices = [0,50,200]

for i in range(len(gs_s)):
    g =gs_s[i]
    ax = f.add_subplot(gs[g[0], g[1]])

    drifter_data.scatter_position_with_labels_flat(ax, labels_xi, colbar=False, size=0.5, 
                                                   t=t_indices[i], cmap = 'tab20', cbarticks = range(-1,np.max(labels_xi)+1),
                                                   norm=norm)
    
    ax.set_title(panel_labels[i] + ' t = ' + str(t_indices[i]), size=10)
    ax.tick_params(labelsize=8)
      
    #axis ticks
    if i==0 or i==1: 
        plt.yticks(np.arange(-2000,4000,2000), np.arange(-2,4,2))
        ax.set_ylabel("y (1000 km)")
    else: plt.yticks([])
    
    if i>0: 
        plt.xticks(np.arange(0,25000,5000), np.arange(0,25,5))
        ax.set_xlabel("x (1000 km)")
    else: plt.xticks([])

f.savefig("./figures/bj_optics_clustering_r_" + str(r) + "_100xi_" + str(int(xi*100)) + "K_" + str(K_embedding), dpi=300)