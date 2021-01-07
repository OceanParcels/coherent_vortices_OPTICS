"""
"Ordering of trajectories reveals hierarchical finite-time coherent sets in Lagrangian particle data"
David Wichmann, Christian Kehl, Henk A. Dijkstra, and Erik van Sebille
questions to d.wichmann@uu.nl

Create matrix with minimum distances between particles for the Agulhas flow.
Used for the creation of the network of Padbergh-Gehle and Schneide 2017
"""

import numpy as np
from particle_and_network_classes import trajectory_data

particle_data = trajectory_data.from_npz("Agulhas_particles.npz", 
                                          domain_name = "agulhas_domain",
                                          t_select = range(21))

mindist_matrix = particle_data.compute_mindist_matrix()
np.save("Agulhas_mindist_matrix", mindist_matrix)
