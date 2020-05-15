"""
Classes to process drifter data:
    - Class domain: specify the region of interest (here North Atlantic).
    - Class trajectory data: handle lon/lat/time drifter data and construct networks from them
    - Class undirected_network: network analysis, mostly spectral clustering, of an undirected network
    - Class bipartite_network: handles spectral clustering of bipartite networks
Notes:
    - Naming of x and y coordinates is geographic (lon, lat), as the code was originally set up for geo applications
    - If applied to other regions than north atlantic, adjust the data set and the domain
"""

import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from mpl_toolkits.basemap import Basemap
from scipy.sparse import coo_matrix
from datetime import datetime, timedelta
import scipy.sparse
import networkx as nx
from sklearn.cluster import KMeans
from netCDF4 import Dataset
import matplotlib.animation as animation

class domain(object):
    """
    Class containing the domain, partition definition and domain limits for plotting.
    """
    
    def __init__(self, minlon, maxlon, minlat, maxlat,
                 minlon_plot, maxlon_plot, minlat_plot, 
                 maxlat_plot, parallels, meridians):
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat
        
        #Domain limits for plotting
        self.minlon_plot = minlon_plot
        self.maxlon_plot = maxlon_plot
        self.minlat_plot = minlat_plot
        self.maxlat_plot = maxlat_plot
        self.parallels = parallels
        self.meridians = meridians
        
    def setup_grid(self, d_x, d_y):
        self.d_x = d_x
        self.d_y = d_y

        #Align grid with spacing    +    
        self.maxlat_grid = (self.maxlat // self.d_y + 1) * self.d_y
        self.minlat_grid = (self.minlat // self.d_y) * self.d_y
        self.maxlon_grid = (self.maxlon // self.d_x + 1) * self.d_x
        self.minlon_grid = (self.minlon // self.d_x) * self.d_x
        
        #Edges of the grid
        self.Lats_edges = np.linspace(self.minlat_grid, self.maxlat_grid,
                                      int((self.maxlat_grid - self.minlat_grid)/d_y)+1) 
        self.Lons_edges = np.linspace(self.minlon_grid, self.maxlon_grid,
                                      int((self.maxlon_grid - self.minlon_grid)/d_x)+1) 
        #Centres of the grid
        self.Lons_centered=np.diff(self.Lons_edges)/2+self.Lons_edges[:-1]
        self.Lats_centered=np.diff(self.Lats_edges)/2+self.Lats_edges[:-1]

        #Meshgrid used for plotting
        self.lon_bins_2d, self.lat_bins_2d = np.meshgrid(self.Lons_edges, self.Lats_edges)   
    
    
    """
    Methods to set up standard grids. 
    - domain_edges: parameters defined externally or through checking the data. Should contain all data of interest    
    """    
    
    @classmethod
    def south_atlantic_domain(cls, domain_edges):    
        (minlon, maxlon, minlat, maxlat) = domain_edges
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-20., 55, -55., -15.)
        (parallels, meridians) = ([-45,-35,-25,80], [-5,15,35])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)
    
    
    @classmethod
    def agulhas_domain(cls, initial_domain_edges):    
        (minlon, maxlon, minlat, maxlat) = (-180., 180., -90., 90.)
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = initial_domain_edges
        (parallels, meridians) = ([-35,-25], [-20,0,20])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)
    
    @classmethod
    def global_domain(cls):    
        (minlon, maxlon, minlat, maxlat) = (-180., 180., -90., 90.)
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-180., 180., -90., 90.)
        (parallels, meridians) = ([20,40,60,80],[-70,-40,-10,20,50])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)
    
    @classmethod
    def north_atlantic_domain(cls, domain_edges):    
        (minlon, maxlon, minlat, maxlat) = domain_edges
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (-100., 40., 0., 85.)
        (parallels, meridians) = ([20,40,60,80],[-70,-40,-10,20,50])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)

    @classmethod
    def double_gyre_domain(cls, domain_edges):
        (minlon, maxlon, minlat, maxlat) = domain_edges
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (0., 2., 0., 1.)
        (parallels, meridians) = ([],[])
        return cls(minlon, maxlon, minlat, maxlat, 
                   minlon_plot, maxlon_plot, minlat_plot, maxlat_plot,
                   parallels, meridians)

    @classmethod
    def bickley_jet_domain(cls, initial_domain_edges):
        r0 = 6371.
        (minlon, maxlon, minlat, maxlat) = (0., np.pi * r0, -5000, 5000)
        (minlon_plot, maxlon_plot, minlat_plot, maxlat_plot) = (0., np.pi * r0, -2500, 2500)
        (parallels, meridians) = ([],[])
        return cls( minlon,  maxlon,  minlat,  maxlat, 
                    minlon_plot,  maxlon_plot,  minlat_plot, maxlat_plot,
                   parallels, meridians)
        

class trajectory_data(object):
    """
    Class to
    - handle drifter data
    - compute the networks used for clustering
    - plot data, e.g. a field over North Atlantic Ocean
    """
    
    def __init__(self, drifter_longitudes=[], drifter_latitudes=[], 
                 drifter_time=[], drifter_id=[], time_0 =  datetime(1900, 1, 1, 0, 0), 
                 time_interval=1, set_nans = True, delete_nans = False,
                 domain_type = "north_atlantic_domain", initial_domain_edges = None):
        """
        Parameters:
        - drifter_longitudes: format is len-N list of numpy arrays, each array containing the data of a trajectory.
        Similar for drifter_latitudes, drifter_time (seconds since time_0).
        - drifter_id: len-N list of drifter ID's. ID is not used at any point in the analysis.
        - time_0: reference time. time unit is in seconds since then.
        - time_interval: this is the time step between two measurements. 
        - set_nans: Some drifters have erroneous data. If True, this data is set to nan.
        - delete nans: if True, the driters containing nan values are deleted
        - : specifies parameters for self.domain. Options: "north_atlantic_domain"
            "double_gyre_domain", "bickley_jet_domain"
        """

        self.drifter_longitudes = drifter_longitudes
        self.drifter_latitudes = drifter_latitudes
        self.drifter_time = drifter_time
        self.drifter_id = drifter_id
        self.N = len(drifter_longitudes)
        self.time_interval = time_interval
        self.time_0 = time_0

        if set_nans:
            self.set_nans(delete_nans=delete_nans)
            self.nans_set = True
        else:
            self.nans_set = False

        
        #Find domain of particle trajectories
        # if initial_domain_edges == None:
        #     print('Domain edges not specified. Computing them from drifter data.')
        #     i=0
        #     minlat = np.nanmin(self.drifter_latitudes[i])
        #     maxlat = np.nanmax(self.drifter_latitudes[i])
        #     minlon = np.nanmin(self.drifter_longitudes[i])
        #     maxlon = np.nanmax(self.drifter_longitudes[i])
            
        #     for i in range(1, self.N):
        #         minlat = np.nanmin([minlat, np.nanmin(self.drifter_latitudes[i])])
        #         maxlat = np.nanmax([maxlat, np.nanmax(self.drifter_latitudes[i])])
        #         minlon = np.nanmin([minlon, np.nanmin(self.drifter_longitudes[i])])
        #         maxlon = np.nanmax([maxlon, np.nanmax(self.drifter_longitudes[i])])
        #     domain_edges = (minlon, maxlon, minlat, maxlat)
        #     print('Domain edges: ', domain_edges)
        
        self.initial_domain_edges = initial_domain_edges
        self.domain = getattr(domain, domain_type)(initial_domain_edges = initial_domain_edges)
        self.domain_type = domain_type
        print('Trajectory data with ' + str(self.N) + ' drifters is set up')
    
    
    @classmethod
    def from_nc(cls, pdir, fname, initial_domain_edges, tload=None, grid_start=0, 
                grid_end=40, domain_type='global_domain'):
        """
        Load data from netcdf particle output files
        - pdir: directory of files
        - fname: file name in pdir to loop over different files containing different particle sets
        - tload: vector of times to be loaded
        - Ngrids: number of different grid output files
        """

        print('Loading data from files: ', pdir + fname)
        
        i = grid_start #Load data from first grid-array
        print('Load grid no: ', i)
        pfile   = pdir + fname + str(i)+'.nc'     
        data    = Dataset(pfile,'r')
        lons    = data.variables['lon'][:,tload]
        lats    = data.variables['lat'][:,tload]
        z       = data.variables['z'][:,tload]
        
        for i in range(grid_start + 1,grid_end): #Load data from other grid-arrays
            print('Load grid no: ', i)
            pfile   = pdir + fname + str(i)+'.nc'  
            data    = Dataset(pfile,'r')
            lons    = np.vstack((lons, data.variables['lon'][:,tload]))
            lats    = np.vstack((lats, data.variables['lat'][:,tload]))
            z       = np.vstack((z, data.variables['z'][:,tload]))

        
        return cls(drifter_longitudes = lons, drifter_latitudes=lats, drifter_time = [],
                   time_interval=5, set_nans = False, domain_type = domain_type, 
                   initial_domain_edges = initial_domain_edges)
        
    
    @classmethod
    def from_npz(cls, filename, set_nans = False, delete_nans=False,
                 time_interval=1, n_step=1,
                 domain_type = "north_atlantic_domain", initial_domain_edges=None):
        """
        Load data from .npz file:
            - filename has to contain: drifter_longitudes, drifter_latitudes, drifter_time (empty for model flows)
            - time_interval_days: the days between different data points of the
            - n_step: can restrict to every n_step'th data point. E.g. if data is 
            6-hourly (time_interval=0.25 days) and n_step=4, then trajectory_data object will be daily
            - domain_edges: edges of the full domain, i.e. for partition. Defined in self.domain.
        """
        
        data = np.load(filename, allow_pickle = True)
        lon = data['drifter_longitudes']
        lat = data['drifter_latitudes']
        time = data['drifter_time']
        lon = [lo[::n_step] for lo in lon]
        lat = [la[::n_step] for la in lat]
        time = [t[::n_step] for t in time]
        
        return cls(drifter_longitudes = lon, drifter_latitudes=lat, drifter_time = time,
                   time_interval=time_interval*n_step, set_nans = set_nans,
                   delete_nans = delete_nans, domain_type = domain_type, initial_domain_edges = initial_domain_edges)
        
    
    def set_nans(self, constrain_to_initial_domain=False, delete_nans=False):
        """
        Function to either identify data points as erronous (ocean drifters), or 
        constrain data to certain domain, e.g. if particles leave the domain
        - constrain_to_domain: if true, all data outside of self.domain_edges set to nan
        """
        
        #The drifter data contains some empty data (where lon / lat are >600).
        # if self.domain_type == 'agulhas_domain':
        #     print('Setting erroneous lon/lat  data points to nan')
        #     # n_data_errors = 0
        #     remove_indices = []
        #     for i in range(self.N):
        #         # n_data_errors += len(self.drifter_longitudes[i][self.drifter_longitudes[i]>180])
        #         # self.drifter_latitudes[i][self.drifter_longitudes[i]>180]=np.nan
        #         # self.drifter_longitudes[i][self.drifter_longitudes[i]>180]=np.nan
        #         if len(np.argwhere(np.isnan(self.drifter_longitudes[i])))>=N_nans:
        #             remove_indices += [i]
        
        # print('Number of trajectories containing at least ' + str(N_nans) 
        #        + ' nan values: '+ str(len(remove_indices)))
        

        if constrain_to_initial_domain:
            print('Constraining particles to domain')
            (minlon, maxlon, minlat, maxlat) = self.initial_domain_edges
            remove_indices = []
            for i in range(self.N):
                if i%10000 ==0: print(str(i) + ' / ' + str(self.N))
                lon0 = self.drifter_longitudes[i][0]
                lat0 = self.drifter_latitudes[i][0]
                
                if lon0<minlon or lon0>maxlon or lat0<minlat or lat0>maxlat:
                    remove_indices += [i]
        
        
        print(len(remove_indices))
        if delete_nans:
            self.drifter_latitudes = np.delete(self.drifter_latitudes, remove_indices, axis=0)
            self.drifter_longitudes = np.delete(self.drifter_longitudes, remove_indices, axis=0)
            self.drifter_time = np.delete(self.drifter_time, remove_indices, axis=0)
            self.N = len(self.drifter_longitudes)        
    
    
        # if constrain_to_domain:
        #     (minlon, maxlon, minlat, maxlat) = self.domain_edges
        #     for i in range(self.N):
        #         self.drifter_latitudes[i][self.drifter_latitudes[i]>maxlat]=np.nan
        #         self.drifter_latitudes[i][self.drifter_latitudes[i]<minlat]=np.nan
        #         self.drifter_longitudes[i][self.drifter_longitudes[i]>maxlon]=np.nan
        #         self.drifter_longitudes[i][self.drifter_longitudes[i]<minlon]=np.nan
                
        self.nans_set = True
    
    
    def set_max_length(self, max_length):
        """
        For ocean drifters: set maximum length for each trajectory and append 
        remaining parts as new trajectories
        """        
        M = self.N
        i=0
        while i<M:
            if len(self.drifter_longitudes[i])>max_length:
                self.drifter_longitudes.append(self.drifter_longitudes[i][max_length:])
                self.drifter_latitudes.append(self.drifter_latitudes[i][max_length:])
                self.drifter_longitudes[i] = self.drifter_longitudes[i][:max_length]
                self.drifter_latitudes[i] = self.drifter_latitudes[i][:max_length]
                M = M + 1
            i = i+1
        self.N = len(self.drifter_longitudes)
    
    
    # def set_min_length(self, min_length):
    #     """
    #     For ocean drifters: set minimum length for each trajectory and remove 
    #     the shorter trajectories
    #     """     
    #     M = self.N
    #     i=0
    #     while i<M:
    #         if len(self.drifter_longitudes[i])<min_length:
    #             self.drifter_longitudes.pop(i)
    #             self.drifter_latitudes.pop(i)
    #             M = M - 1
    #             i = i - 1
    #         i = i + 1
        
    #     self.N = len(self.drifter_longitudes)
    
    
    def start_end_times(self):
        """
        Compute start and end times of drifters in datetime format
        """
        self.start_times = [timedelta(seconds = self.drifter_time[i][0]) + self.time_0 for i in range(len(self.drifter_time))]
        self.end_times = [timedelta(seconds = self.drifter_time[i][-1]) + self.time_0 for i in range(len(self.drifter_time))]
        self.trajectory_lengths = np.array([(self.end_times[i] - self.start_times[i]).days for i in range(len(self.end_times))])


    def compute_C(self, T_index):
        rows = []
        columns = []
        values = []
        
        for i in range(self.N):
            if len(self.symbolic_sequence[i])>T_index:
                if not np.isnan(self.symbolic_sequence[i][T_index]):
                    columns += [self.symbolic_sequence[i][T_index]]
                    rows += [i]                
        values = [1.] * len(columns)        
        return coo_matrix((values, (rows, columns)), shape=(self.N, self.n_horizontal)).tocsr()
        

    def set_discretizing_values(self, bin_size):
        """
        Set partition size
        """
        if len(bin_size) == 1: self.d_x, self.d_y = bin_size[0], bin_size[0]
        else: self.d_x, self.d_y = bin_size[0], bin_size[1]
        self.domain.setup_grid(d_x = self.d_x, d_y = self.d_y)
        self.n_lons  = int((self.domain.maxlon_grid - self.domain.minlon_grid)/self.d_x)
        self.n_lats  = int((self.domain.maxlat_grid - self.domain.minlat_grid)/self.d_y)
        self.n_horizontal   = self.n_lons * self.n_lats        
    
    
    def restrict_to_subset(self, indices):
        """
        Limit drifter data to a subset, labelled by 'indices'. Used to remove data points.
        This is not applied to drifter_time, as for the model flows drifter_time is not defined!
        """
        self.drifter_longitudes = [self.drifter_longitudes[i] for i in indices]
        self.drifter_latitudes = [self.drifter_latitudes[i] for i in indices]
        self.N = len(indices)
        
            
    def coords_to_matrixindex2D(self, coord_2D):
        """
        Function to compute the symbol for a certain binning, from a given (lon, lat) coordinate
        """
        
        (lon, lat) = coord_2D
        if np.isnan(lon) or np.isnan(lat): 
            return np.nan
        else:
            index_2D    = int((lat - self.domain.minlat_grid)//self.d_y * self.n_lons + (lon - self.domain.minlon_grid)//self.d_x)
            return index_2D
    

    def compute_symbolic_sequences(self, bin_size, dt=1, 
                                   trajectory_segments = False):
        """
        - Function to compute, for each drifter, the symbolic sequence given the 
        cell size of 'bin_size' and time interval of dt.
        - if trajectory_segments = True, trajectories are segmented into smaller parts. 
        E.g. for a trajectory with length 10, dt=2, we produce two trajectories, one
        with indices [0,2,4,...] and one with [1,3,5,...].
        """
        
        self.set_discretizing_values(bin_size)
        self.dt = dt
        dt_indices = int(dt / self.time_interval)
        
        if trajectory_segments:
            self.symbolic_sequence = [] 
            for k in range(dt_indices):
                self.symbolic_sequence += [np.array([self.coords_to_matrixindex2D((lo,la)) for lo, la in 
                                          zip(self.drifter_longitudes[i][k::dt_indices],
                                              self.drifter_latitudes[i][k::dt_indices])]) for i in range(self.N)]
        else:
            self.symbolic_sequence = [np.array([self.coords_to_matrixindex2D((lo,la)) for lo, la in 
                                          zip(self.drifter_longitudes[i][::dt_indices],
                                              self.drifter_latitudes[i][::dt_indices])]) for i in range(self.N)]
    
    
    def compute_initial_symbols(self, d_deg):
        """
        Only compute initial symbols and distribution for plotting
        """
        self.set_discretizing_values(d_deg)
        self.initial_symbols = np.array([self.coords_to_matrixindex2D((lo[0],la[0])) for lo, la in zip(self.drifter_longitudes,
                                                                                              self.drifter_latitudes)])
        unique_symbols, counts = np.unique(self.initial_symbols, return_counts=True)
        d = np.zeros(self.n_horizontal)
        d[unique_symbols] = counts
        self.initial_distribution = d
                

    def scatter_initial_position_flat(self, ax, indices):
        """
        Scatter initial positions (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        
        if indices == []: indices = range(len(self.drifter_longitudes))
        
        lonplot = [self.drifter_longitudes[i][0] for i in indices]
        latplot = [self.drifter_latitudes[i][0] for i in indices]
        # lonplot = [lon_reduced[i][0] for i in range(len(lon_reduced))]
        # latplot = [lat_reduced[i][0] for i in range(len(lat_reduced))]  
        
        ax.scatter(lonplot, latplot, s=2, alpha = 0.5)



    def scatter_initial_position(self, ax, indices=None):
        """
        Scatter initial positions (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        # m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        # m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        lon_reduced = self.drifter_longitudes[indices]
        lat_reduced = self.drifter_latitudes[indices]
        lonplot = [lon_reduced[i][0] for i in range(len(lon_reduced))]
        latplot = [lat_reduced[i][0] for i in range(len(lat_reduced))]        
        xs, ys = m(lonplot, latplot)
        ax.scatter(xs, ys, s=2, alpha = 0.5)


   

    def scatter_trajectories_geo(self, ax, indices):
        """
        For earth surface
        Scatter whole trajectories (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        lon_reduced = self.drifter_longitudes[indices]
        lat_reduced = self.drifter_latitudes[indices]
        
        lonplot = lon_reduced[0]
        latplot = lat_reduced[0]
        
        for i in range(1, len(lon_reduced)):
            lonplot = np.append(lonplot, lon_reduced[i])
            latplot = np.append(latplot, lat_reduced[i])
        
        xs, ys = m(lonplot, latplot)
        ax.scatter(xs, ys, s=2, alpha = 0.5)
        
        
    def scatter_trajectories_flat(self, ax, indices):
        """
        For flat domain
        Scatter whole trajectories (of a subset labelled by indices).
        If indices is None, then take whole data set
        """
        
        if indices == None: indices = range(len(self.drifter_longitudes))
        
        lon_reduced = [self.drifter_longitudes[i] for i in indices]
        lat_reduced = [self.drifter_latitudes[i] for i in indices]
        
        lonplot = lon_reduced[0]
        latplot = lat_reduced[0]
        
        for i in range(1, len(lon_reduced)):
            lonplot = np.append(lonplot, lon_reduced[i])
            latplot = np.append(latplot, lat_reduced[i])
        
        ax.plot(lonplot, latplot, 'o--', markersize=5)
        ax.plot(lonplot[0], latplot[0], 'o', color='r', markersize=10)
        
        ax.set_xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
        ax.set_ylim([self.domain.minlat_plot, self.domain.maxlat_plot])
        

    def animate_particles_geo(self, labels, filename='movie', animation_time=10, cmap=None, norm=None):
        
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
            plt.title('day ' + str(t*self.time_interval))
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


    def scatter_position_with_labels_geo(self, ax, labels=None, size = 4, cmap=None, norm=None,
                                             cbar=False, cbarlabel = None, cbarticks = None,
                                             random_shuffle = True, t=0):
        """
        For earth surface: Scatter initial positions with color map given by labels
        - ax: pyplot axis
        - labels: colors
        - size: markersize in plot
        - cmap: colormap
        - norm: norm for colorbar
        - cbar: Plots cbar for True
        - cbarlabel: title of cbar
        - cbarticks: ticks on cbar
        - random_shuffle: if True, randomly shuffle particles and labels such that 
        not one color completely covers the other
        - t: particle time for plotting
        """
        
        
        lon_plot = np.array([lo[t] for lo in self.drifter_longitudes])
        lat_plot = np.array([lo[t] for lo in self.drifter_latitudes])
        # if labels==[]: labels = ['k'] * len(lon_plot)
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()
        m.fillcontinents(color='dimgray')
        
        if random_shuffle:
            indices = np.array(range(len(lon_plot)))
            np.random.shuffle(indices)
            lon_plot = lon_plot[indices]
            lat_plot = lat_plot[indices]
            labels = labels[indices]
        
        xs, ys = m(lon_plot, lat_plot)
        if cmap == None:
            p = ax.scatter(xs, ys, s=size, c=labels, alpha=0.6)   
        else:
            p = ax.scatter(xs, ys, s=size, c=labels, cmap = cmap, norm=norm, alpha=0.6)   
        # ax.set_xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
        # ax.set_ylim([self.domain.minlat_plot, self.domain.maxlat_plot])
        if cbar:
            if cbarlabel == None:
                cbar = plt.colorbar(p, shrink=.6, aspect=25, ticks=cbarticks, orientation="horizontal")
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=25, orientation="horizontal", ticks=cbarticks)
                cbar.set_label(cbarlabel, size=9) 
                # cbar.ax.set_ylabel(cbarlabel)
        
    
    
        # def scatter_position_with_labels_geo(self, ax, labels=None, size = 4, cmap=None, norm=None,
        #                                      cbar=False, cbarlabel = None, cbarticks = None,
        #                                      random_shuffle = True, t=0):
    
    def scatter_position_with_labels_flat(self, ax, labels, size=8, cmap=None, norm=None,
                                          colbar=False, cbarlabel = None, cbarticks = None,
                                          random_shuffle = True, alpha=1, t=0):
        """
        For flat space: Scatter initial positions with color map given by labels
        - ax: pyplot axis
        - labels: colors
        - size: markersize in plot
        - cmap: colormap
        - norm: norm for colorbar
        - cbar: Plots cbar for True
        - cbarlabel: title of cbar
        - cbarticks: ticks on cbar
        - random_shuffle: if True, randomly shuffle particles and labels such that 
        not one color completely covers the other
        - alpha: transparency of markers
        - t: particle time for plotting
        """
        
        lon_plot = np.array([lo[t] for lo in self.drifter_longitudes])
        lat_plot = np.array([la[t] for la in self.drifter_latitudes])
        # if labels==[]: labels = ['k'] * len(lon_plot)
        
        if random_shuffle:
            indices = np.array(range(len(lon_plot)))
            np.random.shuffle(indices)
            lon_plot = lon_plot[indices]
            lat_plot = lat_plot[indices]
            labels = labels[indices]
        
        if cmap == None:
            p = ax.scatter(lon_plot, lat_plot, s=size, c=labels, alpha=alpha)
        else:
            p = ax.scatter(lon_plot, lat_plot, s=size, c=labels, cmap = cmap, norm=norm, alpha=alpha)
        
        if colbar:
            if cbarlabel == None:
                cbar = plt.colorbar(p, shrink=.6, aspect=15, ticks=cbarticks, orientation="horizontal")
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=15, orientation="horizontal", ticks=cbarticks)
                cbar.set_label(cbarlabel, size=9) 

        ax.set_xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
        ax.set_ylim([self.domain.minlat_plot, self.domain.maxlat_plot])



    def plot_discretized_distribution_geo(self, ax, v, cmap = None, norm=None, land = False, 
                                      title = 'notitle', colbar=True, 
                                      cbartitle = None, logarithmic = False,
                                      cbar_orientation = 'horizontal', extent='neither'):
        """
        Plot binned 2D field on earth surface
        - ax: axis object
        - v: field to plot
        - cmap: colormap
        - norm: norm for color bar
        - land: land is plotted if True
        - title: title of figure
        - colbar: colorbar is plotted if True
        - cbartitle: title of colorbar
        - logarithmic: logarithmic cbar if True
        - cbar_orientation: placement of colorbar
        - extent: for capping colorbar
        """

        #convert to 3-dimensional vector if needed
        if v.ndim == 1: d2d = v.reshape(self.n_lats, self.n_lons)
        else: d2d = v
        
        #plotting
        if cmap == None: cmap = 'plasma'
        if norm == None: norm= colors.Normalize(vmin=np.ma.min(d2d), vmax=np.ma.max(d2d))
        
        m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
                    llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
        m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
        m.drawcoastlines()

        if land: m.fillcontinents(color='dimgray')
        xs, ys = m(self.domain.lon_bins_2d, self.domain.lat_bins_2d)
        if logarithmic:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap,  norm=colors.LogNorm())
        else:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap, norm=norm)# , vmin=-20, vmax=1)

        if colbar: 
            if cbar_orientation == 'horizontal':
                cbar = plt.colorbar(p, shrink=.6, aspect=10, orientation="horizontal", extend=extent)
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=40, extend=extent)
            cbar.ax.tick_params(labelsize=9) 
            if cbartitle is not None:
                cbar.set_label(cbartitle, size=9)  
        
    
    def plot_discretized_distribution_flat(self, ax, v, cmap = None, norm=None, land = False, 
                                      title = 'notitle', colbar=True, 
                                      cbartitle = None, logarithmic = False,
                                      cbar_orientation = 'horizontal'):
        """
        See plot_discretized_distribution_gee for parameter description
        """

        #convert to 3-dimensional vector if needed
        if v.ndim == 1: d2d = v.reshape(self.n_lats, self.n_lons)
        else: d2d = v
        
        #plotting
        if cmap == None: cmap = 'plasma'
        if norm == None: norm= colors.Normalize(vmin=np.ma.min(d2d), vmax=np.ma.max(d2d))
        
        xs = self.domain.lon_bins_2d
        ys = self.domain.lat_bins_2d
        
        if logarithmic:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap,  norm=colors.LogNorm())
        else:
            p = ax.pcolormesh(xs, ys, d2d, rasterized=True, cmap =  cmap, norm=norm)# , vmin=-20, vmax=1)
        
        ax.set_xlim([self.domain.minlon_plot, self.domain.maxlon_plot])
        ax.set_ylim([self.domain.minlat_plot, self.domain.maxlat_plot])

        if colbar: 
            if cbar_orientation == 'horizontal':
                cbar = plt.colorbar(p, shrink=.6, aspect=10, orientation="horizontal")
            else:
                cbar = plt.colorbar(p, shrink=.6, aspect=40)
            cbar.ax.tick_params(labelsize=9) 
            if cbartitle is not None:
                cbar.set_label(cbartitle, size=9) 
        
        
        
    # def distance_on_sphere(self, i1, i2):
    #     """
    #     Function to return a vector of distances (km) for two drifters with indices i1 and i2.
    #     Assumed is that the vectors are defined for the same times.
    #     """
    #     lats1 = self.drifter_latitudes[i1]
    #     lons1 = self.drifter_longitudes[i1]
    #     lats2 = self.drifter_latitudes[i2]
    #     lons2 = self.drifter_longitudes[i2]
    
    #     r = 6371.
    #     a = np.pi/180.
    #     arg = np.sin(a * lats1) * np.sin(a * lats2) + np.cos(a * lats1) * np.cos(a * lats2) * np.cos(a * (lons1-lons2))
        
    #     for i in range(len(arg)):
    #         if np.abs(arg[i]) > 1-1e-5: arg[i]=1
        
    #     d = r * np.arccos(arg)
            
    #     return d
        

    # def distance_euclidean(self, i1, i2):
    #     """
    #     Function to return a vector of distances (km) for two drifters with indices i1 and i2.
    #     Assumed is that the vectors are defined for the same times.
    #     """
    #     lats1 = self.drifter_latitudes[i1]
    #     lons1 = self.drifter_longitudes[i1]
    #     lats2 = self.drifter_latitudes[i2]
    #     lons2 = self.drifter_longitudes[i2]    
    #     d = np.nanmin(np.sqrt((lats1- lats2)**2 + (lons1 - lons2)**2))        
    #     return d


    # def distance_on_sphere_time_incoherent(self, i1, i2):
    #     """
    #     Function to return a vector of distances (km) for two drifters with indices i1 and i2.
    #     The distances are only computed for the temporal overlap of the drifter data. If they
    #     do not overlap, nan is returned.
        
    #     """
    #     time1, time2 = self.drifter_time[i1], self.drifter_time[i2]
        
    #     min_time = np.max([np.min(time1), np.min(time2)])
    #     max_time = np.min([np.max(time1), np.max(time2)])
    
    #     interval_1 = np.argwhere((time1 >= min_time) & (time1 < max_time))[:,0]
    #     interval_2 = np.argwhere((time2 >= min_time) & (time2 < max_time))[:,0]
    #     assert(len(interval_1) == len(interval_2))
        
    #     lons1, lats1 = self.drifter_longitudes[i1][interval_1], self.drifter_latitudes[i1][interval_1]
    #     lons2, lats2 = self.drifter_longitudes[i2][interval_2], self.drifter_latitudes[i2][interval_2]
    #     time1, time2 = self.drifter_time[i1][interval_1], self.drifter_time[i2][interval_2]
    #     assert(np.all(time1 - time2 ==0.)) #Check that the time intervals are the same
    #     assert(len(lons1) == len(lons2))
        
    #     if len(lons1)==0: #no overlap
    #         return np.nan
    #     else:
    #         r = 6371.
    #         a = np.pi/180.
    #         arg = np.sin(a * lats1) * np.sin(a * lats2) + np.cos(a * lats1) * np.cos(a * lats2) * np.cos(a * (lons1-lons2))
            
    #         for i in range(len(arg)):
    #             if (np.abs(arg[i]) - 1.)>1e-5: arg[i]=1
            
    #         d = r * np.arccos(arg)
    #         return d
        
    
    # def compute_minimum_distances(self, distance_function='distance_on_sphere'):
    #     """
    #     Function to compute NxN network containing the minimum distance of two drifters.
    #     - distance_function: 'distance_on_sphere_time_incoherent', 'distance_on_sphere'
    #     """
        
    #     distance_function = getattr(self, distance_function)
        
    #     if not self.nans_set:
    #         self.set_nans()
        
    #     A = np.empty((self.N, self.N))
    #     print(A.shape)
    #     for i in range(self.N):
    #         if i%50 ==0: print(str(i) + " / " + str(self.N))
    #         for j in range(i, self.N):
    #             A[i,j] = np.nanmin(distance_function(i,j))
                
    #     return A


    # def compute_network_PadbergGehle(self, distance_function='distance_on_sphere', 
    #                                  eps = 1):
    #     """
    #     Function to compute NxN network containing the minimum distance of two drifters.
    #     - distance_function: 'distance_on_sphere_time_incoherent', 'distance_on_sphere'
    #     """
        
    #     distance_function = getattr(self, distance_function)
        
    #     if not self.nans_set:
    #         self.set_nans()
        
    #     A = np.empty((self.N, self.N))
    #     print(A.shape)
    #     for i in range(self.N):
    #         if i%50 ==0: print(str(i) + " / " + str(self.N))
    #         for j in range(i, self.N):
    #             A[i,j] = np.nanmin(distance_function(i,j))
                
    #     return A


    # def compute_T(self, distance_function='distance_on_sphere'):
    #     """
    #     Function to compute NxN network containing the minimum distance of two drifters.
    #     - distance_function: 'distance_on_sphere_time_incoherent', 'distance_on_sphere'
    #     """
        
    #     distance_function = getattr(self, distance_function)
        
    #     if not self.nans_set:
    #         self.set_nans()
        
    #     A = np.empty((self.N, self.N))
    #     print(A.shape)
    #     for i in range(self.N):
    #         if i%50 ==0: print(str(i) + " / " + str(self.N))
    #         for j in range(i, self.N):
    #             A[i,j] = np.nanmean(distance_function(i,j))
                
    #     self.T_mean_distances = A


    # def compute_P(self, trajectory_segments = False):
    #     """
    #     Function to compute the transition matrix P, and the cell connection matrix, C.
    #     Note that the domain is chosen such that drifters that leave it have negative bin index,
    #     which we use to define it as Markovian drifter or not.
    #     """
        
    #     print('Computing transition matrix')
        
        
    #     symbolic_sequence = self.symbolic_sequence
        
    #     if trajectory_segments:
    #         initial_indices = []
    #         final_indices = []
    #         for i in range(len(symbolic_sequence)):
    #             initial_indices += list(symbolic_sequence[i][:-1])
    #             final_indices += list(symbolic_sequence[i][1:])
    #     else:
    #         initial_indices = [s[0] for s in symbolic_sequence]
    #         final_indices = [s[-1] for s in symbolic_sequence]
        
    #     initial_indices = np.array(initial_indices)
    #     final_indices = np.array(final_indices)
        
    #     # indices = (initial_indices>0) & (~np.isnan(final_indices))
    #     # initial_indices = initial_indices[indices]
    #     # final_indices = final_indices[indices]
        
    #     print('Computing transition matrix')
    #     transitions = np.ones(len(initial_indices))
        
    #     W = coo_matrix((transitions, (initial_indices, final_indices)), shape=(self.n_horizontal, self.n_horizontal)).tocsr()
    #     d = np.array(sparse.csr_matrix.sum(W, axis=1))[:,0] #initial particle distribution
    #     D_inv = sparse.diags([1./di if di>0 else 0. for di in d])
    #     P =  (W.transpose().dot(D_inv)).transpose()
        
    #     self.initial_indices = initial_indices
    #     self.final_indices = final_indices
    #     self.W = W
    #     self.d = d
    #     self.P = P

        
    # def surface_area_field(self):
    #     """
    #     Function to return the surface area (km2) of the chosen cells as a vector
    #     """
    #     a = 6371.
    #     diff_sin_ = np.diff(np.sin(self.domain.Lats_edges * np.pi/180.))
    #     A_lat = a**2 * diff_sin_ * (self.d_deg * np.pi / 180.)
    #     area = np.array([list(A_lat)])
    #     A_horizontal = np.array([list(np.repeat(area, self.n_lons, axis=0).transpose())])        
    #     return A_horizontal[0]  





    # def trajectories_density(self, indices, nmax=None):
    #     """
    #     Function to return a density field defined by all points on the trajectory
    #     """
    #     seq = self.symbolic_sequence
        
    #     if indices == None: indices = range(len(self.drifter_longitudes))
        
    #     s = np.array(seq[indices[0]][:nmax]).astype(int)
        
    #     for i in indices[1:]:
    #         s = np.append(s, seq[i][:nmax].astype(int))
        
    #     s = s[s>=0]
    #     s_unique, s_counts = np.unique(s[~np.isnan(s)], return_counts=True)
        
    #     d = np.zeros(self.n_horizontal)
    #     d[s_unique] = s_counts
        
    #     self.d_cluster = d


    # def scatter_trajectories_with_labels(self, ax, labels, random_sample=100, seed=None):
    #     """
    #     Scatter entire trajectories with color map given by labels.
    #     - random_sample: size of a random subsample to accelerate plotting
    #     """
    #     np.random.seed(seed)
    #     indices = np.random.randint(0, len(self.drifter_longitudes), size=random_sample)
    #     labels = labels[indices]
    #     lon_reduced = self.drifter_longitudes[indices]
    #     lat_reduced = self.drifter_latitudes[indices]
        
    #     lonplot = lon_reduced[0]
    #     latplot = lat_reduced[0]
    #     c = np.array([labels[0]] * len(lon_reduced[0]))
        
    #     for i in range(1, len(lon_reduced)):
    #         lonplot = np.append(lonplot, lon_reduced[i])
    #         latplot = np.append(latplot, lat_reduced[i])
    #         c = np.append(c, np.array([labels[i]] * len(lon_reduced[i])))
        
    #     m = Basemap(projection='mill',llcrnrlat=self.domain.minlat_plot, urcrnrlat=self.domain.maxlat_plot, 
    #                 llcrnrlon=self.domain.minlon_plot, urcrnrlon=self.domain.maxlon_plot, resolution='c')
    #     m.drawparallels(self.domain.parallels, labels=[True, False, False, True], linewidth=1., size=9, color='lightgray')
    #     m.drawmeridians(self.domain.meridians, labels=[False, False, False, True], linewidth=1., size=9, color='lightgray')
    #     m.drawcoastlines()
    #     m.fillcontinents(color='dimgray')
        
    #     indices_plot = np.random.shuffle(np.array(range(len(lonplot))))
    #     xs, ys = m(lonplot, latplot)
    #     p = ax.scatter(xs[indices_plot], ys[indices_plot], s=2, c=c[indices_plot],  alpha = 0.1)
    #     plt.colorbar(p, shrink=.6, aspect=25)
        


class bipartite_network(object):
    """
    Class to  handle bipartite network analysis.    
    """
    
    def __init__(self, B):
        self.adjacency_matrix = B
        self.N = B.shape[0]
        self.M = B.shape[1]        


    def stochastic_complement_adjacency_matrix(self, space = 'X'):
        """
        Return adjacency matrix for stochastic complement, i.e. two-step random walk
        """
        if space == 'X':
            q = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=0))[0,:]
            PI_q_inv = sparse.diags([1/qi if qi!=0 else 0 for qi in q])
            return (self.adjacency_matrix.dot(PI_q_inv)).dot(self.adjacency_matrix.transpose())
        
        elif space == 'Y': 
            p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
            PI_p_inv = sparse.diags([1/pi if pi!=0 else 0 for pi in p])
            return (self.adjacency_matrix.transpose()).dot(PI_p_inv).dot(self.adjacency_matrix)


    def projection_adjacency_matrix(self, space = 'X'):
        """
        Return adjacency matrix for projection, i.e. GG^T (or G^TG)
        """
        if space == 'X':
            return self.adjacency_matrix.dot(self.adjacency_matrix.transpose())
        
        elif space == 'Y': 
            return self.adjacency_matrix.transpose().dot(self.adjacency_matrix)

 
    def stochastic_complement_laplacian_spectrum(self, K=20):     
        print('Computing the SVD of the stochastic complementation.') 
        p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        q = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=0))[0,:]
        PI_p_inv_sqrt = sparse.diags([1/np.sqrt(pi) if pi!=0 else 0 for pi in p])
        PI_q_inv_sqrt = sparse.diags([1/np.sqrt(qi) if qi!=0 else 0 for qi in q])
        B_hat = PI_p_inv_sqrt.dot(self.adjacency_matrix).dot(PI_q_inv_sqrt)        
        u, s, vt = sparse.linalg.svds(B_hat, K)
        indices = np.argsort(s)[::-1]    
        u=u[:,indices]
        s = s[indices]
        vt = vt[indices,:]        
        return [PI_p_inv_sqrt.dot(u), s, PI_q_inv_sqrt.dot(vt.transpose())]
    
    def projection_laplacian_spectrum_ShiMalik(self, K=20):
        print('Computing the SVD of the projection (Shi Malik).')
        p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix.T, axis=1))[:,0]
        p = self.adjacency_matrix.dot(p)
        PI_p_inv_sqrt = sparse.diags([1/np.sqrt(pi) if pi!=0 else 0 for pi in p])
        B_hat = PI_p_inv_sqrt.dot(self.adjacency_matrix)
        u, s, vt = sparse.linalg.svds(B_hat, K)
        indices = np.argsort(s)[::-1]    
        u=u[:,indices]
        s = s[indices]
        vt = vt[indices,:]
        return [PI_p_inv_sqrt.dot(u), s, vt.transpose()]
        

    def projection_laplacian_spectrum_normalized_cut_Ng(self, K=20):
        print('Computing the SVD of the projection (Ng Jordan).')
        p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix.T, axis=1))[:,0]
        p = self.adjacency_matrix.dot(p)
        PI_p_inv_sqrt = sparse.diags([1/np.sqrt(pi) if pi!=0 else 0 for pi in p])
        B_hat = PI_p_inv_sqrt.dot(self.adjacency_matrix)
        u, s, vt = sparse.linalg.svds(B_hat, K)
        indices = np.argsort(s)[::-1]
        u=u[:,indices]
        s = s[indices]
        vt = vt[indices,:]
        
        return [u, s, vt.transpose()]
        

    

    def out_degrees(self):
        p = np.array(sparse.csr_matrix.sum(self.adjacency_matrix.T, axis=1))[:,0]
        p = self.adjacency_matrix.dot(p)
        q = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        q = self.adjacency_matrix.T.dot(q)        
        return [p, q]


def construct_dendrogram(networks):
    """
    Construct a dendrogram from a hierarchical list of 'undirected_network' objects
    - networks: list of network groups. len(networks[0])=1, len(networks[1])=2, etc.
    """
    K = len(networks)
    network_labels = [[nw.cluster_label for nw in networks[i]] for i in range(K)]
    linkage_labels = [[nw.cluster_label for nw in networks[i]] for i in range(K)]
    original_obs = [[1 for _ in networks[i]] for i in range(K)]
    n = len(network_labels[-1])
    Z = np.zeros((n-1,4)) #linkage function
    
    for i in range(1,len(network_labels))[::-1]:
        network_label1 = network_labels[i][-1]
        network_label2 = network_labels[i][-2]
        label = np.min([network_label1, network_label2])
        i_merged = np.argwhere(network_labels[i-1]==label)[0][0]
        new_label = n + n-1-i
        old_label = linkage_labels[i-1][i_merged]
        for j in range(i):
            linkage_labels[j] = [l if l!=old_label else new_label for l in linkage_labels[j]]
            original_obs
        Z[n-1-i][0] = linkage_labels[i][-1]
        Z[n-1-i][1] = linkage_labels[i][-2]
        
    ncut = []
    for i in range(len(networks)):
        r = np.sum([networks[i][j].rho for j in range(len(networks[i]))])
        ncut.append(len(networks[i])-r)
    
    ncut=np.array(ncut)
    Z[:,2]= np.max(ncut[1:]) - ncut[1:][::-1]+.2
    return Z




class undirected_network(object):
    """
    Class to handle analysis of undirected networks
    """
    
    def __init__(self, adjacency_matrix, cluster_indices=np.array([]), 
                 cluster_volume=np.array([]), cluster_label = 0):
        
        """
        adjacency_matrix: format sparse.csr_matrix. If it is not symmetric it is symmetrized.
        region_indices: indices corresponding to network domain.
        cluster_volume: vector of volume of the nodes inside the cluster.
        """
        if len(cluster_indices)==0:
            cluster_indices = np.array(range(adjacency_matrix.shape[0]))
        if len(cluster_volume)==0:
            cluster_volume = np.array(sparse.csr_matrix.sum(adjacency_matrix, axis=1))[:,0]
        
        self.adjacency_matrix = adjacency_matrix
        self.cluster_indices = cluster_indices
        self.cluster_volume = cluster_volume
        self.N = adjacency_matrix.shape[0]
        self.cluster_label = cluster_label
        self.rho = np.sum(self.adjacency_matrix)/np.sum(self.cluster_volume)
        assert(len(cluster_indices) == self.adjacency_matrix.shape[0])
        assert(len(cluster_volume) == len(cluster_indices))
        print('Construct undirected network.')
    
    
    def __del__(self):
        print('Adjacency matrix object deleted')


    
    def largest_connected_component(self):

        """
        Determine connected components
        """
        
        print('Find connected components')
        
        G = nx.from_scipy_sparse_matrix(self.adjacency_matrix, create_using = nx.Graph())
        components = np.array(list(nx.connected_components(G)))
        component_lengths = np.array([len(s) for s in components])
        component_inds = np.argsort(component_lengths)[::-1]
        components_sorted = components[component_inds]
        component_lengths = np.array([len(c) for c in components_sorted])
        
        print('Lengths of components (>1)')
        print(component_lengths[component_lengths>1])
        
        #Largest component
        inds = list(components_sorted[0])
        sub_adjacency_matrix = self.adjacency_matrix[inds, :][:, inds]
        sub_cluster_indices = self.cluster_indices[inds]
        sub_cluster_volume = self.cluster_volume[inds]
        sub_cluster_label = self.cluster_label
        return undirected_network(sub_adjacency_matrix, sub_cluster_indices, 
                                               sub_cluster_volume, sub_cluster_label)


    def compute_spectrum_L_symmnorm(self, K=20, plot=False):
        """
        Comput eigenvectors for clustering from symmetric nocmralized Laplacian
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

        return w, v
    
    
    def compute_spectrum_L(self, K=20, plot=False):
        """
        Comput eigenvectors for clustering from symmetric nocmralized Laplacian
        """
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D = scipy.sparse.diags(d)
        L = D - self.adjacency_matrix
        print('Computing spectrum of graph Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        if plot:
            plt.plot(w, 'o')
            plt.title('Eigenvalues of graph Laplacian')
            plt.grid(True)
            plt.show()

        return w, v
    

    def compute_spectrum_L_randomwalk(self, K=20, plot=False):
        """
        Comput eigenvectors for clustering from symmetric nocmralized Laplacian
        """
        d = np.array(sparse.csr_matrix.sum(self.adjacency_matrix, axis=1))[:,0]
        D_sqrt_inv = scipy.sparse.diags([1./np.sqrt(di) if di!=0 else 0 for di in d ])
        L = sparse.identity(self.N) - (D_sqrt_inv.dot(self.adjacency_matrix)).dot(D_sqrt_inv)
        print('Computing spectrum of random walk Laplacian')
        w, v = sparse.linalg.eigsh(L, k=K, which = 'SM')
        inds = np.argsort(w)
        w = w[inds]
        v = v[:,inds]
        
        if plot:
            plt.plot(w, 'o')
            plt.title('Eigenvalues of symmetric normalized Laplacian')
            plt.grid(True)
            plt.show()

        return 1-w, D_sqrt_inv.dot(v)
   

    def drho_split(self, indices_1, indices_2):
        """
        If we propose to split a cluster, this returns the changes in the coherence ratio for a split into
        indices_1 and indices_2
        """
        cluster_volume_1 = np.sum(self.cluster_volume[indices_1])
        cluster_volume_2 = np.sum(self.cluster_volume[indices_2])
        stays_in_1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_1])
        stays_in_2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_2])        
        return stays_in_1 / cluster_volume_1 + stays_in_2 / cluster_volume_2 - self.rho


    def rho_split(self, indices_1, indices_2):
        """
        If we propose to split a cluster, this returns the changes in the coherence ratio for a split into
        indices_1 and indices_2
        """
        cluster_volume_1 = np.sum(self.cluster_volume[indices_1])
        cluster_volume_2 = np.sum(self.cluster_volume[indices_2])
        stays_in_1 = np.sum(self.adjacency_matrix[indices_1, :][: ,indices_1])
        stays_in_2 = np.sum(self.adjacency_matrix[indices_2, :][: ,indices_2])        
        return stays_in_1 / cluster_volume_1 + stays_in_2 / cluster_volume_2


    def hierarchical_clustering_ShiMalik(self, K, plots=False):
        """
        Implementation of hierarchical clustering according to Shi & Malik 2000.
        At each iteration, one cluster is added, minimizing the global NCut. We implement this
        by computing the increase in rho and choose the maximum increase.
        """
        networks = {}
        networks[0] = [self]

        for i in range(1,K):
            print('Level: ', i)
            
            optimal_drhos = []
            optimal_cutoffs = []
            
            for j in range(len(networks[i-1])):
                nw = networks[i-1][j]
                if nw.N<100: 
                    optimal_drhos.append(np.nan)
                    optimal_cutoffs.append(np.nan)
                    continue
        
                nw.compute_laplacian_spectrum()
                V_fiedler = nw.Lsym_eigenvectors[:,1]
                c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
                
                drhos = []
                for c in c_range:
                
                    indices_1 = np.argwhere(V_fiedler<=c)[:,0]
                    indices_2 = np.argwhere(V_fiedler>c)[:,0]
                    drhos.append(nw.drho_split(indices_1, indices_2))
                    
                drhos = np.array(drhos)
                if plots:
                    plt.plot(c_range, drhos)
                    plt.yscale('log')
                    plt.grid(True)
                    plt.title(r'$\Delta \rho_{global}$ for different cutoffs. Network' + str(i) + str(j))
                    plt.show()
                cutoff_opt = c_range[np.nanargmax(drhos)]
                print('Choosing as cutoff: ', str(cutoff_opt))
                
                optimal_drhos.append(np.nanmax(drhos))
                optimal_cutoffs.append(cutoff_opt)
            
            i_cluster = np.nanargmax(optimal_drhos)
            print('Splitting cluster ', i_cluster+1)
            cutoff_cluster = optimal_cutoffs[np.nanargmax(optimal_drhos)]
            nw_to_split = networks[i-1][i_cluster]
            V_fiedler = nw_to_split.Lsym_eigenvectors[:,1]
            indices_1 = np.argwhere(V_fiedler<=cutoff_cluster)[:,0]
            indices_2 = np.argwhere(V_fiedler>cutoff_cluster)[:,0]
            
            if len(indices_1)<len(indices_2):
                ind_ = indices_1.copy()
                indices_1= indices_2.copy()
                indices_2 = ind_
            
            adjacency_matrix_1 = nw_to_split.adjacency_matrix[indices_1, :][:, indices_1]
            adjacency_matrix_2 = nw_to_split.adjacency_matrix[indices_2, :][:, indices_2]
            cluster_indices_1 = nw_to_split.cluster_indices[indices_1]
            cluster_indices_2 = nw_to_split.cluster_indices[indices_2]
            cluster_volume_1 = nw_to_split.cluster_volume[indices_1]
            cluster_volume_2 = nw_to_split.cluster_volume[indices_2]
            
            cluster_label_1 = nw_to_split.cluster_label
            
            old_labels = [nw.cluster_label for nw in networks[i-1]]
            
            cluster_label_2 = np.max(old_labels)+1
            
            network_children = [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                            undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
            
            networks[i] = networks[i-1].copy()
            networks[i].pop(i_cluster)
            networks[i] += network_children #append in the back
        
        self.clustered_networks = networks
        
    
    
    def hierarchical_clustering_individual(self, K, rho_min, plots=True):
        networks = {}
        networks[0] = [self]

        for i in range(1,K):
            print('Level: ' + str(i))
            networks[i] = []
            for nw in networks[i-1]:
                if nw.rho>rho_min:
                    w, v = nw.compute_laplacian_spectrum(K=10)
                    V_fiedler = v[:,1]
                    
                    c_range = np.linspace(np.min(V_fiedler), np.max(V_fiedler), 100)[1:]
                
                    rhos = []
                    for c in c_range:
                    
                        indices_1 = np.argwhere(V_fiedler<=c)[:,0]
                        indices_2 = np.argwhere(V_fiedler>c)[:,0]
                        rhos.append(nw.rho_split(indices_1, indices_2))
                
                    if plots:
                        plt.plot(c_range, rhos)
                        plt.yscale('log')
                        plt.grid(True)
                        plt.title(r'$\Delta \rho_{local}$ for different cutoffs.')
                        plt.show()
                    
                    cutoff_opt = c_range[np.nanargmax(rhos)]
                    indices_1 = np.argwhere(V_fiedler<=cutoff_opt)[:,0]
                    indices_2 = np.argwhere(V_fiedler>cutoff_opt)[:,0]
                    
                    if len(indices_1)<len(indices_2):
                        ind_ = indices_1.copy()
                        indices_1= indices_2.copy()
                        indices_2 = ind_
                    
                    
                    adjacency_matrix_1 = nw.adjacency_matrix[indices_1, :][:, indices_1]
                    adjacency_matrix_2 = nw.adjacency_matrix[indices_2, :][:, indices_2]
                    cluster_indices_1 = nw.cluster_indices[indices_1]
                    cluster_indices_2 = nw.cluster_indices[indices_2]
                    cluster_volume_1 = nw.cluster_volume[indices_1]
                    cluster_volume_2 = nw.cluster_volume[indices_2]
                    
                    cluster_label_1 = nw.cluster_label
                    
                    old_labels = [nw.cluster_label for nw in networks[i-1]]
                    
                    cluster_label_2 = np.max(old_labels)+1
                    
                    networks[i] += [undirected_network(adjacency_matrix_1, cluster_indices_1, cluster_volume_1, cluster_label_1), 
                                    undirected_network(adjacency_matrix_2, cluster_indices_2, cluster_volume_2, cluster_label_2)]
                    
                else:
                    networks[i] += [nw]
            
        self.clustered_networks = networks
                    