"""
Ordering of trajectories reveals hierarchical finite-time coherent
sets in Lagrangian particle data: detecting Agulhas rings in the
South Atlantic Ocean
------------------------------------------------------------------------------
David Wichmann, Christian Kehl, Henk A. Dijkstra, Erik van Sebille

"""

"""
Main library to handle trajectory data and network analysis.

Classes:
    - Class domain: contains the region of interest used to for plotting
    - Class trajectory_data: handle lon/lat/time drifter data and construct networks from them
    - Class undirected_network: network analysis, mostly spectral clustering, of an undirected network
Notes:
    - Naming of x and y coordinates is geographic (lon, lat), as the code was originally set up for geo applications
    - If applied to other regions, adjust the data set and the domain
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.basemap import Basemap
import scipy.sparse
import matplotlib.animation as animation
from scipy.spatial.distance import pdist, squareform


class domain(object):
    """
    Class containing the plotting domain
    """
    
    def __init__(self, minlon_plot, maxlon_plot, minlat_plot, 
                 maxlat_plot, parallels, meridians, domain_type):
        
        #Domain limits for plotting
        self.minlon_plot = minlon_plot
        self.maxlon_plot = maxlon_plot
        self.minlat_plot = minlat_plot
        self.maxlat_plot = maxlat_plot
        
        #For geographic plots labelling of parallels and meridians
        self.parallels = parallels
        self.meridians = meridians
        self.domain_type = domain_type
        
    @classmethod
    def agulhas_domain(cls):    
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-30., 20, -40., -20.)
        # (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-50., 40, -60., -10.)
        (parallels, meridians) = ([-35,-25], [-20,0,20])
        return cls(minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians, domain_type='agulhas')
    
    @classmethod
    def bickley_jet_domain(cls):
        r0 = 6371.
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (0., np.pi * r0, -3000, 3000)
        (parallels, meridians) = ([],[])
        return cls(minlon_plot,  maxlon_plot,  minlat_plot, maxlat_plot,
                   parallels, meridians, domain_type='bickley_jet')


class trajectory_data(object):
        
    def __init__(self, drifter_longitudes=[], drifter_latitudes=[],
                 domain_name = "north_atlantic_domain"):
        """
        Parameters:
        - drifter_longitudes: format is len-N list of numpy arrays, each array containing the data of a trajectory.
        Similar for drifter_latitudes, drifter_time (seconds since time_0).
        - domain_name: specifies parameters for self.domain. Options: "agulhas_domain"
            "bickley_jet_domain"
        """
        self.drifter_longitudes = drifter_longitudes
        self.drifter_latitudes = drifter_latitudes
        self.N = len(drifter_longitudes)
        self.T = len(drifter_longitudes[0])
        self.domain_name = domain_name
        self.domain = getattr(domain, domain_name)()
        print('Trajectory data with ' + str(self.N) + ' trajectories of length ' 
              + str(self.T) +' is set up')
    
    @classmethod
    def from_npz(cls, filename, domain_name, t_select=None, n_select = None):
        """
        Load data from .npz file:
            - filename has to contain: drifter_longitudes, drifter_latitudes, 
            drifter_time (empty for model flows)
            - domain_name: see __init__
        """
        data = np.load(filename, allow_pickle = True)
        lon = data['drifter_longitudes']
        lat = data['drifter_latitudes']
        
        if t_select is not None:
            lon = lon[:, t_select]
            lat = lat[:, t_select]
        if n_select is not None:
            lon = lon[n_select, :]
            lat = lat[n_select, :]
        
        return cls(drifter_longitudes = lon, drifter_latitudes=lat, 
                   domain_name=domain_name)
    
    def restrict_to_subset(self, indices):
        """
        Limit drifter data to a subset, labelled by 'indices'. Used to remove data points to generate incomplete
        data set for the model flow. This is not applied to drifter_time, as for the model flows drifter_time is not defined.
        """
        lon = [self.drifter_longitudes[i] for i in indices]
        lat = [self.drifter_latitudes[i] for i in indices]
        return trajectory_data(drifter_longitudes = lon, drifter_latitudes=lat, 
                               domain_name=self.domain_name)
    
                
    def scatter_position_with_labels(self, ax, labels, size = 4, cmap=None, norm=None,
                                         cbar=False, cbartitle="", random_shuffle = True, 
                                         alpha=0.6, t=0, vmax=None):
        """
        For earth surface: Scatter positions at certain time t with color map given by labels
        - ax: pyplot axis
        - labels: colors
        - size: markersize in plot
        - cmap: colormap
        - norm: norm for colorbar
        - land: True if land is to be filled in plot
        - cbar: Plots cbar for True
        - random_shuffle: if True, randomly shuffle particles and labels such that 
        not one color completely covers the other
        - t: particle time for plotting
        - alpha: transparency of scatter points
        """
        
        lon_plot = np.array([lo[t] for lo in self.drifter_longitudes])
        lat_plot = np.array([lo[t] for lo in self.drifter_latitudes])
        
        if random_shuffle:
            indices = np.array(range(len(lon_plot)))
            np.random.shuffle(indices)
            lon_plot = lon_plot[indices]
            lat_plot = lat_plot[indices]
            labels = labels[indices]
        
        if self.domain.domain_type == 'agulhas':
        
            m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                        llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
            m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
            m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
            m.drawcoastlines()
            m.fillcontinents(color='dimgray')
            xs, ys = m(lon_plot, lat_plot)
        elif self.domain.domain_type == 'bickley_jet':
            xs, ys = lon_plot, lat_plot
        
        if cmap == None: p = ax.scatter(xs, ys, s=size, c=labels, alpha=alpha)   
        else:  p = ax.scatter(xs, ys, s=size, c=labels, cmap = cmap, norm=norm, 
                              alpha=alpha, vmax=vmax)   

        if cbar: 
            cbar = plt.colorbar(p, shrink=.8, aspect=10, orientation='horizontal', extend='max')
            cbar.set_label(cbartitle)

        if self.domain.domain_type == 'bickley_jet':        
            ax.set_xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
            ax.set_ylim([self.domain.minlat_plot, self.domain.maxlat_plot])


    def animate_particles_geo(self, labels, filename='movie', 
                              animation_time=None, cmap=None, norm=None):
        
        if animation_time == None:
            animation_time = self.T

        fig = plt.figure(figsize=(10, 6))
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        lons = [self.drifter_longitudes[i][0] for i in range(self.N)]
        lats = [self.drifter_latitudes[i][0] for i in range(self.N)]
        
        print(len(lons))
        xs, ys = m(lons, lats)
        scat = m.scatter(xs, ys, s=.5, c=labels, cmap=cmap, norm=norm)

        def animate(t):
            if t%10 == 0:  print('time: ', t)
            lons = [self.drifter_longitudes[i][t] for i in range(self.N)]
            lats = [self.drifter_latitudes[i][t] for i in range(self.N)]
            scat.set_offsets(np.matrix(m(lons,  lats)).transpose())
            plt.title('Day: ' + str(t * 5))
            return scat
        

        anim = animation.FuncAnimation(fig, animate, frames=range(0,animation_time), blit=False)
        anim.save(filename + '.mp4', fps=5, extra_args=['-vcodec', 'libx264'])   


    def animate_particles_flat(self, labels, filename='movie', animation_time=10, cmap=None, norm=None):
            
        fig = plt.figure(figsize=(8, 3))
        plt.xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
        plt.ylim([self.domain.minlat_plot, self.domain.maxlat_plot])
        
        lons = [self.drifter_longitudes[i][0] for i in range(self.N)]
        lats = [self.drifter_latitudes[i][0] for i in range(self.N)]
        
        print(len(lons))
        scat = plt.scatter(lons, lats, s=7, c=labels, cmap=cmap, norm=norm)

        def animate(t):
            if t%10 == 0:  print('time: ', t)
            lons = [self.drifter_longitudes[i][t] for i in range(self.N)]
            lats = [self.drifter_latitudes[i][t] for i in range(self.N)]
            scat.set_offsets(np.matrix((lons,  lats)).transpose())
            plt.title('day ' + str(t*self.time_interval))
            return scat
        
        anim = animation.FuncAnimation(fig, animate, frames=range(0,animation_time), blit=False)
        anim.save(filename + '.mp4', fps=10, extra_args=['-vcodec', 'libx264'])   


    def compute_mindist_matrix(self):
        """
        Function to compute NxN network containing the minimum distance of two drifters.
        """
        X_full = self.compute_data_embedding()

        X = X_full[:,::self.T]
        print (X.shape)
        D_min = pdist(X, metric='euclidean')
        
        for t in range(1, self.T):
            print(str(t) + " / " + str(self.T-1))
            X = X_full[:,t::self.T]
            D = pdist(X, metric='euclidean')
            D_min = np.minimum(D_min, D)
        
        return squareform(D_min)


    def compute_data_embedding(self, MDS=False):
        
        r0 = 6371.
        if self.domain.domain_type == 'agulhas':
            a = np.pi/180.
            x = np.array([r0 * np.cos(a * la) * np.cos(a * lo)  for lo, la in zip(self.drifter_longitudes, self.drifter_latitudes)])
            y = np.array([r0 * np.cos(a * la) * np.sin(a * lo)  for lo, la in zip(self.drifter_longitudes, self.drifter_latitudes)])
            z = np.array([r0 * np.sin(a * la)  for la in self.drifter_latitudes])
            X = np.hstack((x, y, z))
        elif self.domain.domain_type == 'bickley_jet':
            x = np.array([np.pi * r0 / 2 * np.cos(2 * lo/r0) for lo in self.drifter_longitudes])
            y = np.array([np.pi * r0 / 2 * np.sin(2 * lo/r0) for lo in self.drifter_longitudes])
            z = np.array([la for la in self.drifter_latitudes])
            X = np.hstack((x,y,z))
        if MDS == False:
            return X
        else:
            print("Computing classical MDS embedding (takes a while!)")
            D = pdist(X, metric='euclidean')
            n = X.shape[0]
            del X
            D2 = squareform(D**2)
            del D
            print("D**2 computed")
            
            H = np.eye(n) - np.ones((n, n))/n
            K = -H.dot(D2).dot(H)/2
            print("K computed")
            del D2
            vals, vecs = np.linalg.eigh(K)
            del K
            print("Done!")
            indices   = np.argsort(vals)[::-1]
            vals = vals[indices]
            vecs = vecs[:,indices]
            indices_relevant, = np.where(vals > 0)
            Xbar  = vecs[:,indices_relevant].dot(np.diag(np.sqrt(vals[indices_relevant])))
            return vals, Xbar


class undirected_network(object):
    """
    Class to handle analysis of undirected networks
    """
    
    def __init__(self, adjacency_matrix):
        
        """
        - adjacency_matrix: format sparse.csr_matrix. If it is not symmetric it is symmetrized.
        """
        self.adjacency_matrix = adjacency_matrix
        self.N = adjacency_matrix.shape[0]
        print('Construct undirected network.')
        
    def __del__(self):
        print('Adjacency matrix object deleted')


    def compute_laplacian_spectrum(self, K=20, plot=False):
        """
        Compute eigenvectors for clustering from symmetric nocmralized Laplacian
        """
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags([1./np.sqrt(di) if di!=0 else 0 for di in d ])
        L = sparse.identity(self.N) - (D_sqrt_inv.dot(self.adjacency_matrix)).dot(D_sqrt_inv)
        print('Computing spectrum of symmetric normalized Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        if plot:
            plt.plot(w, 'o')
            plt.title('Eigenvalues of symmetric normalized Laplacian')
            plt.grid(True)
            plt.show()
        
        return w, D_sqrt_inv.dot(v)