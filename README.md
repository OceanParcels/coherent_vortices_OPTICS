# Code for the paper "Ordering of trajectories reveals hierarchical finite-time coherent sets in Lagrangian particle data"
David Wichmann, Christian Kehl, Henk A. Dijkstra, and Erik van Sebille

Questions to d.wichmann@uu.nl

# Trajectory data

## Bickley jet
Execute create_bickley_jet_trajectories.py

## Agulhas
The trajectories are available at ZENODOLINK. The trajectories are a subset of those used for our previous publication, see the assiciated github repository at https://github.com/OceanParcels/near_surface_microplastic

# Methods

## Network computation
Execute Agulhas_compute_mindist_matrix.py. This computes a matrix with the minimum distances between all particles.

## Compute OPTICS results
Execute clustering_trajectories_optics.py, for the selected option. The script particle_and_network_classes.py contains a few classes to handle the trajectory data and plotting.
