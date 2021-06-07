#!/usr/bin/python
#######
#  ______   ________       _       ____    ____   ______   
# |_   _ \ |_   __  |     / \     |_   \  /   _|.' ____ \  
#   | |_) |  | |_ \_|    / _ \      |   \/   |  | (___ \_| 
#   |  __'.  |  _| _    / ___ \     | |\  /| |   _.____`.  
#  _| |__) |_| |__/ | _/ /   \ \_  _| |_\/_| |_ | \____) | 
# |_______/|________||____| |____||_____||_____| \______.' 
#                                                         
#######
# Project Numerics, COSSE Programme 2021
# Dylan Everingham, Sebastian Myrbäck,
# 	Carsten van de Kamp, Sergi Andreu
# 25.05.21
#
#######

#######
# Dependencies
#######
import numpy as np
import matplotlib.pyplot as plt

#######
# get_mesh_1D
#
# Generates a 1D mesh of a specified resolution, formatted as the standard
# nodes, elems and faces matrices
#
# arguments:
#	N (int) : spacial resolution of mesh, i.e. number of nodes
#	limits (list of float, len=2) : x value of leftmost and rightmost nodes
# 	plot_mesh (bool) : flag indicating if you would like a plot of the mesh
#
# returns:
#	nodes (list of float) : x coordinates of mesh nodes
#	elems (list of [int,int]) : list of elements, which are each defined by
#		two indices into the nodes list
#	faces (list of [int,int]) : list of faces, which are each defined by
#		an index into the nodes list followed by an index which indicates
#		which boundary this face is a member of
#######
def get_mesh_1D(N, limits=[0,1], plot_mesh=False):

	# Generate nodes, elems, faces
	nodes = np.linspace(limits[0],limits[1],N)
	elems = np.array([[n,n+1] for n in range(N-1)])
	faces = np.array([[0,0], [N-1,1]])

	# Do mesh plot
	if (plot_mesh):
		plt.figure(figsize=(6,0.5))
		plt.yticks([])
		plt.scatter(nodes, np.zeros(nodes.shape))
		plt.show()

	return [nodes, elems, faces]