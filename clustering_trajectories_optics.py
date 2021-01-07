"""
<<<<<<< HEAD
Ordering of trajectories reveals hierarchical finite-time coherent
sets in Lagrangian particle data: detecting Agulhas rings in the
South Atlantic Ocean
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

"""

"""
Clustering of trajectories with OPTICS.
=======
"Ordering of trajectories reveals hierarchical finite-time coherent sets in Lagrangian particle data"
David Wichmann, Christian Kehl, Henk A. Dijkstra, and Erik van Sebille
questions to d.wichmann@uu.nl

Clustering of trajectories with OPTICS for the different embeddings
>>>>>>> d28c0a80eb8835edb816c9959a4c08f34b1993a6
"""

import numpy as np
import matplotlib.pyplot as plt
from particle_and_network_classes import trajectory_data, undirected_network
from sklearn.cluster import KMeans, OPTICS
import matplotlib
import matplotlib.colors
from sklearn.cluster import  cluster_optics_xi, cluster_optics_dbscan
import scipy.sparse as sparse

#parameters
r0 = 6371.

parameters = {}

parameters["BJ_direct_embedding_random"] = {"data set": "bickley_jet_trajectories.npz", # trajectory data set
                                     "domain name": "bickley_jet_domain", # domain
                                     "embedding method": "direct embedding", # embedding method
                                     "t_select": range(0,410,50),
                                     "n_select": np.random.randint(0, 12000, 2000), # choose 2,000 random trajectories
                                     "KMeans": False, # if KMeans is performed
                                     "MinPts": 15, # MinPts parameter for OPTICS
                                     "OPTICS": True, # If OPTICS is performed
                                     "optics_params": [["DBSCAN", 18000]],
                                     "ylims": [4000,30000], # y-limit for reachability plots
                                     "t_indices": [3, 6], # times (index) for plots
                                     "t_labels": ["15 days", "30 days"], # time labels for plots
                                     "markersize": 15, # marker size for scatter plots
                                     "plot_embedding": False, # if embedding (first two axes) is plotted
                                     "plot reachabilities": False,
                                     "animation": False,
                                     "filename": "./figures/BJ_direct_embedding_random2000" # filename of figure
                                     }


parameters["BJ_direct_embedding"] = {"data set": "bickley_jet_trajectories.npz", # trajectory data set
                                     "domain name": "bickley_jet_domain", # domain
                                     "embedding method": "direct embedding", # embedding method
                                     "t_select": range(0, 410, 10), # take daily output
                                     "n_select": None, # if we reduce to a subset of trajectories
                                     "KMeans": False, # if KMeans is performed
                                     "MinPts": 80, # MinPts parameter for OPTICS
                                     "OPTICS": True, # If OPTICS is performed
                                     "optics_params": [["DBSCAN", 58000], # OPTICS clustering results
                                                       ["DBSCAN", 38000],
                                                       ["DBSCAN", 15000]],
                                     "ylims": [9000,80000], # y-limit for reachability plots
                                     "t_indices": [15, 30], # times (index) for plots
                                     "t_labels": ["15 days", "30 days"], # time labels for plots
                                     "markersize": 0.3, # marker size for scatter plots
                                     "plot_embedding": False, # if embedding (first two axes) is plotted
                                     "plot reachabilities": False,
                                     "animation": False,
                                     "filename": "./figures/BJ_direct_embedding" # filename of figure
                                     } 



parameters["BJ_MDS"] = {"data set": "bickley_jet_trajectories.npz",
                                    "domain name": "bickley_jet_domain",
                                    "embedding method": "MDS",
                                    "MDS_dim": 2, # embedding dimension for classical MDS
                                    "t_select": range(0,410,10),
                                    "n_select": None,
                                    "KMeans": True,
                                    "Kmeans_clusters": 8,
                                    "OPTICS": True,
                                    "MinPts": 80,
                                    "optics_params": [["DBSCAN", 1000]],
                                    "ylims": [0,10000],
                                    "t_indices": [15, 30],
                                    "t_labels": ["15 days", "30 days"],
                                    "markersize": 0.3,
                                    "plot_embedding": True,
                                    "plot reachabilities": False,
                                    "animation": False,
                                    "filename": "./figures/BJ_cMDS"}


parameters["Agulhas_direct_embedding"] = {"data set": "Agulhas_particles.npz",
                                     "domain name": "agulhas_domain",
                                     "embedding method": "direct embedding",
                                     "t_select": range(21), #long time for animation
                                     "n_select": None,
                                     "KMeans": False,
                                     "Kmeans_clusters": 40,
                                     "MinPts": 100,
                                     "OPTICS": True,
                                     "optics_params": [["DBSCAN", 850], 
                                                       ["DBSCAN", 550],
                                                       ["xi", 0.025]],
                                     "ylims": [400, 1000],
                                     "t_indices": [0, 20],
                                     "t_labels": ["0 days", "100 days"],
                                     "markersize": 0.3,
                                     "plot_embedding": False,
                                     "plot reachabilities": True,
                                     "animation": True,
                                     "filename": "./figures/Agulhas_direct_embedding"}


parameters["Agulhas_MDS_embedding"] = {"data set": "Agulhas_particles.npz",
                                    "domain name": "agulhas_domain",
                                    "embedding method": "MDS",
                                    "MDS_dim": 2,
                                    "t_select": range(21),
                                    "n_select": np.random.randint(0, 23821, 12000), # choose 12,000 random trajectories
                                    "KMeans": True,
                                    "Kmeans_clusters": 40,
                                    "OPTICS": True,
                                    "MinPts": 30,
                                    "optics_params": [["DBSCAN", 80]],
                                    "ylims": [0, 500],
                                    "t_indices": [0, 20],
                                    "t_labels": ["0 days", "100 days"],
                                    "markersize": 0.5,
                                    "plot_embedding": True,
                                    "plot reachabilities": False,
                                    "animation": False,
                                    "filename": "./figures/Agulhas_MDS"}

parameters["Agulhas_network"] = {"data set": "Agulhas_particles.npz",
                                    "domain name": "agulhas_domain",
                                    "embedding method": "network",
                                    "mindist_network": "Agulhas_mindist_matrix.npy",
                                    "d": 200,
                                    "t_select": range(21),
                                    "n_select": None,
                                    "K_embedding": 40,
                                    "Kmeans_clusters": 40,
                                    "K_vectors": 50,
                                    "KMeans": True,
                                    "OPTICS": True,
                                    "MinPts": 100,
                                    "optics_params": [["xi", 0.025],
                                                      ["xi", 0.03]],
                                    "ylims": [0, 0.001],
                                    "t_indices": [0, 20],
                                    "t_labels": ["0 days", "100 days"],
                                    "markersize": 0.3,
                                    "plot_embedding": False,
                                    "plot reachabilities": False,
                                    "animation": False,
                                    "filename": "./figures/Agulhas_network"}

configurations = ["BJ_direct_embedding_random",
                  "BJ_direct_embedding", 
                  "BJ_MDS",
                  "Agulhas_direct_embedding",
                  "Agulhas_MDS_embedding",
                  "Agulhas_network"]

c = "Agulhas_direct_embedding"

print("Configuration: ", c)
P = parameters[c]
particle_data = trajectory_data.from_npz(P["data set"],
                                         domain_name = P["domain name"],
                                         t_select = P["t_select"], 
                                         n_select = P["n_select"])


"""
Compute embedding
"""
if P["embedding method"] == "direct embedding":
    X_embedding = particle_data.compute_data_embedding()
    
if P["embedding method"] == "MDS":
    val, X = particle_data.compute_data_embedding(MDS=True)
    X_embedding = X[:,:P["MDS_dim"]]
    f = plt.figure(constrained_layout=True, figsize = (5, 3))
    ax = f.add_subplot()
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title('Eigenvalues of B')
    plt.grid(True)
    ax.plot(range(len(val)), val, 'o', color = "darkslategrey", markersize=2)
    f.savefig(P["filename"] + "_MDSspectrum", dpi=300)

if P["embedding method"] == "network":
    mindist_matrix = np.load(P["mindist_network"])
    M_full = np.zeros(mindist_matrix.shape)
    M_full[mindist_matrix < P["d"]]=1
    M = sparse.csr_matrix(M_full)
    print("Share of non-zero data points:", len(M.data)/(M_full.shape[0]**2))
    A = undirected_network(M)
    val, X = A.compute_laplacian_spectrum(K=P["K_vectors"], plot=True) 
    X_embedding = X[:,1:P["K_embedding"]+1]
    
    f = plt.figure(constrained_layout=True, figsize = (5, 3))
    ax = f.add_subplot()
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(r'Eigenvalues of $L_r$')
    plt.grid(True)
    ax.plot(range(len(val)), 1-val, 'o', color = "darkslategrey", markersize=2)
    f.savefig(P["filename"] + "_networkSpectrum", dpi=300)


print("Embedding " + P["embedding method"] + " computed.")

"""
K-means (for Agulhas for comparison)
"""
if P["KMeans"]:
    labels_kmeans = KMeans(n_clusters=P["Kmeans_clusters"], random_state=0).fit(X_embedding).labels_
    labels_kmeans = labels_kmeans % 20 # if there are many clusters
    bounds = np.arange(-.5,np.max(labels_kmeans)+1.5,1)
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds))
    
    f = plt.figure(constrained_layout=True, figsize = (10, 2))
    gs = f.add_gridspec(1, 2)
    
    panel_labels = ['(a)','(b)']
    gs_s = [[0,0], [0,1]]
    
    for i in range(len(gs_s)):
        g =gs_s[i]
        ax = f.add_subplot(gs[g[0], g[1]])
    
        particle_data.scatter_position_with_labels(ax, labels_kmeans, cbar=False, size=P["markersize"], 
                                                        t=P["t_indices"][i], cmap = 'tab20',  norm=norm)
        
        ax.set_title(panel_labels[i] + ' t = ' + P["t_labels"][i], size=10)
        
        if P["domain name"] == 'bickley_jet_domain':
            ax.tick_params(labelsize=8)
            if i==0: 
                plt.yticks(np.arange(-2000,4000,2000), np.arange(-2,4,2))
            else: plt.yticks([])
            plt.xticks(np.arange(0,25000,5000), np.arange(0,25,5))
        
    f.savefig(P["filename"] + "_Kmeans", dpi=300)
    

"""
OPTICS
"""

if P["OPTICS"]:
    optics_clustering = OPTICS(min_samples=P["MinPts"], metric="euclidean").fit(X_embedding)
    reachability = optics_clustering.reachability_
    core_distances = optics_clustering.core_distances_
    ordering = optics_clustering.ordering_
    predecessor = optics_clustering.predecessor_
    
    labels = []
    
    for op in P["optics_params"]:
        m, c = op[0], op[1]
        if m == "xi":
            l, _ = cluster_optics_xi(reachability, predecessor, ordering, P["MinPts"], xi=c)
        else:
            l = cluster_optics_dbscan(reachability=reachability,
                                                core_distances=core_distances,
                                               ordering=ordering, eps=c)
        l = np.array([li % 20 if li>=0 else li for li in l])
        labels.append(l) #cmap with 20 colors
    
    norms = []
    for l in labels:
        bounds = np.arange(-.5,np.max(l)+1.5,1)
        norms.append(matplotlib.colors.BoundaryNorm(bounds, len(bounds)))
    
    #Optics results
    f = plt.figure(constrained_layout=True, figsize = (10, 2*len(labels)))
    gs = f.add_gridspec(len(labels), 3)
    
    gs_s = [[[0,1],[0,2]], [[1,1], [1,2]], [[2,1], [2,2]]]
    t_indices = P["t_indices"]
    t_labels = P["t_labels"]
    panel_labels = [['(a) ','(b) ','(c) '],['(d) ', '(e) ','(f) '],['(g) ', '(h) ','(i) ']]
    
    
    #Reachability plots
    for i in range(len(labels)):
        
        Labels_clusters = np.ma.masked_where(labels[i] <0, labels[i], copy=True)
        Labels_noise = ["lightgray"] * len(labels[i])
        Labels_noise =  np.ma.masked_where(labels[i] >= 0, Labels_noise, copy=True)
        
        ax = f.add_subplot(gs[i, 0])
        if "ylims" in P.keys(): ax.set_ylim(P["ylims"])
        ax.scatter(range(len(reachability)), reachability[ordering], c=Labels_clusters[ordering], 
                        cmap = 'tab20', marker="o", s=.1, norm=norms[i])
        ax.scatter(range(len(reachability)), reachability[ordering], c=Labels_noise[ordering], 
                        cmap = 'tab20', marker="o", s=.1, norm=norms[i], alpha=0.5)
        
        if P["optics_params"][i][0] == "xi":
            a, b = r"$\xi$", P["optics_params"][i][1]
        else:
            a, b = r"$\epsilon$", P["optics_params"][i][1]
            ax.axhline(P["optics_params"][i][1], color="k")
        
        if P["optics_params"][i][0] == "DBSCAN":
            cluster_type = "DBSCAN"
        else:
            cluster_type = r"$\xi$"
        
        if P["domain name"] == "bickley_jet_domain":
            if b % 1000 == 0:
                print_b = str(int(b/1000))
                if print_b == "1":
                    print_b = ""
                else:
                    print_b = print_b + r"$\cdot$"
            ax.set_title(panel_labels[i][0] + cluster_type + "-clustering, " +  a + ' = ' + print_b + r"$10^6$ km", size=10)
            ax.set_ylabel(r"$r(p_i)\; (10^3$ km)")
        elif P["domain name"] == "agulhas_domain":
            if P["embedding method"] == "network":
                ax.set_title(panel_labels[i][0] + cluster_type + "-clustering, " +  a + ' = ' + str(b), size=10)
                ax.set_ylabel(r"$r(p_i)$")
            elif P["optics_params"][i][0] == 'xi':
                ax.set_title(panel_labels[i][0] + cluster_type + "-clustering, " +  a + ' = ' + str(b), size=10)
                ax.set_ylabel(r"$r(p_i)\; (km)$")                
            else:
                ax.set_title(panel_labels[i][0] + cluster_type+ "-clustering, " +  a + ' = ' + str(b) + " km", size=10)
                ax.set_ylabel(r"$r(p_i)\; (km)$")
                
        ax.set_xlabel(r"$i$")
        ax.tick_params(labelsize=8)
    
        for t in range(len(t_indices)):
            g =gs_s[i][t]
            ax = f.add_subplot(gs[g[0], g[1]])
            
            
            particle_data.scatter_position_with_labels(ax, Labels_clusters, cbar=False, size=P["markersize"], 
                                                            t=t_indices[t], cmap = 'tab20', norm=norms[i])
            
            particle_data.scatter_position_with_labels(ax, Labels_noise, cbar=False, size=P["markersize"], 
                                                            t=t_indices[t], alpha=.4)
            
            ax.set_title(panel_labels[i][t+1]  + 't = ' + t_labels[t], size=10)
            
            if P["domain name"] == 'bickley_jet_domain':
                ax.tick_params(labelsize=8)
                
                if t==0: plt.yticks(np.arange(-2000,4000,2000), np.arange(-2,4,2))
                else: plt.yticks(np.array([]))
                
                if i == len(labels)-1: plt.xticks(np.arange(0,25000,5000), np.arange(0,25,5))
                else: plt.xticks([])
                
    f.savefig(P["filename"] + "_optics", dpi=300)


"""
Plot embedded trajectories with last added labels in OPTICS
"""
if P["plot_embedding"]:
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9,3), sharey=True)
    
    ax1.scatter(X_embedding[:,0], X_embedding[:,1], c = "lightgrey", 
                marker="o", s=0.008, cmap = 'tab20', norm=norms[-1])
    ax2.scatter(X_embedding[:,0], X_embedding[:,1], c = Labels_clusters, 
                marker="o", s=0.008, cmap = 'tab20', norm=norms[-1])
    ax2.scatter(X_embedding[:,0], X_embedding[:,1], c = Labels_noise, 
                marker="o", s=0.008, cmap = 'tab20', norm=norms[-1])
    ax3.scatter(X_embedding[:,0], X_embedding[:,1], c = labels_kmeans, 
                marker="o", s=0.008, cmap = 'tab20', norm=norms[-1])
    ax3.set_xlabel("dimension 1")
    ax2.set_xlabel("dimension 1")
    ax1.set_xlabel("dimension 1")
    ax1.set_title("(a) No labels")
    if P["domain name"] == "bickley_jet_domain":
        ax2.set_title(r"(b) " + P["optics_params"][i][0] + "-clustering, " +  a + ' = ' + print_b + r"$10^6$ km")
    elif P["domain name"] == "agulhas_domain":
        ax2.set_title(r"(b) " + P["optics_params"][i][0] + "-clustering, " +  a + ' = ' + str(b) + r" km")
    ax3.set_title("(c) k-Means (K=" + str(P["Kmeans_clusters"]) + ")")
    ax1.set_ylabel("dimension 2")
    plt.tight_layout()
    f.savefig(P["filename"] + "_embedding", dpi=300)
    
    
"""
Plot reachabilities
"""
if P["plot reachabilities"]:
    f = plt.figure(constrained_layout=True, figsize = (6, 3))
    ax = f.add_subplot()
    particle_data.scatter_position_with_labels(ax, reachability, cbar=True, size=1,
                                                  t=0, cmap = 'inferno', vmax=1000,
                                                  cbartitle = "reachability value (km)")   
    f.savefig(P["filename"] + "_reachabilities", dpi=300)
    
if P["animation"]:
    particle_data = trajectory_data.from_npz(P["data set"],
                                         domain_name = P["domain name"],
                                         t_select = range(145), 
                                         n_select = P["n_select"])
    particle_data.animate_particles_geo(Labels_clusters, 
                                    filename='movie', 
                                    animation_time=145, 
                                    cmap='tab20', 
                                    norm=norms[0])