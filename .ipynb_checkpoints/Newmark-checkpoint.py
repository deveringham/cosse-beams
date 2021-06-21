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
# Carsten van de Kamp, Sergi Andreu,
# Sebastian Myrbäck, Dylan Everingham
# 17.06.21
#
# Newmark.py
# Classes representing FEM solutions with time discretization using Newmark's method to the 1D beam problem.
#######


#######
# Dependencies
#######

from NumericalSolutions import *

from scipy import integrate
from scipy.sparse import csr_matrix
import numpy as np

from scipy.sparse.linalg import spsolve

from meshes import *


class Newmark:
    
    #The class Newmark lets an initial state (u1, up1, upp1) evolve dynamically using the Newmark method

    def __init__(self, u1=0, up1=0, upp1=0, h=0.1, beta=1/4, gamma=1/2, Me=1,
                 Se=1, f=1):

        # Set initial values
        self.u1 = u1
        self.up1 = up1
        self.upp1 = upp1

        # Set timestep
        self.h = h

        self.uniform = True
        
        if not isinstance(self.h, float):
            self.uniform = False
        
        # Set numerical parameters
        assert beta > 0 and beta <=1/2
        assert gamma > 0 and gamma <=1
        self.beta = beta
        self.gamma = gamma
        
        # Dictionary of arrays containing all the intermediate values
        self.results = {"u": [self.u1], "up": [self.up1],
                        "upp": [self.upp1], "h": [0], "t": [0],
                       "w":[self.u1[:-2][::2]], "wp":[self.u1[:-2][1::2]]}
        
        # Matrices and arrays needed for the solver
        self.Me = Me
        self.Se = Se
        self.f = f
        
        
    def __get_ustars(self, u, up, upp, h):
        ustar = u + h * up + (0.5 - self.beta) * upp * h**2
        ustarp = up + (1 - self.gamma) * upp * h

        return ustar, ustarp
   
    # The equation governing the dynamics is Me(x | mu)¨ + Se(x | mu) - (f | a) = 0
    # we are using u = (x | mu)
    
    
    def __solver(self, ustar):
        # All matrices and vectors are given as numpy arrays
        rhs = self.f - self.Se @ ustar
        lhs = self.Me + self.beta * self.h**2 * self.Se
        
        upp = spsolve(lhs, rhs)
        
        return upp
    
    
    def __get_next(self, ustar, upstar, upp, h):
        unext = ustar + self.beta * upp * h**2
        
        upnext = upstar + self.gamma * upp * h
        
        return unext, upnext
        
    
    def run(self, niterations, log_values = True):
        assert self.uniform
        h = self.h
        
        t = 0
        
        # Initialize the values at the initial b.c.
        u = self.u1
        up = self.up1
        upp = self.upp1
        
        for i in range(niterations):
            
            t += h
            
            ustar, upstar = self.__get_ustars(u, up, upp, h)
            
            # NOTE THAT THE SOLVER DOES NOT NEED ustarp

            upp = self.__solver(ustar)
            
            u, up = self.__get_next(ustar, upstar, upp, h)
            
            w = u[:-2][::2]
            wp = u[:-2][1::2] #In this case, the derivative is wrt the space variable
            
            
            if log_values:
                
                self.results["u"].append(u)
                self.results["up"].append(up)
                self.results["upp"].append(upp)
                self.results["h"].append(h)
                self.results["t"].append(t)
                self.results["w"].append(w)
                self.results["wp"].append(wp)
                
        return u, up, upp, t

    



class Dynamic:
    
    #The class Dynamic generates the necessary initial states and matrices for the beam
    
    def __init__(self, E=1, I=1, N=1, L=1,
                 initial_conditions = {"a":0, "b":0, "QL":0, "M0":0, "ML":0, "a0":0, "aL":0, "q":0}, 
                 parameters = {"a":0, "b":0, "QL":0, "M0":0, "ML":0, "a0":0, "aL":0, "q":0},
                two_sided_support = True):
        
        #Fixed parameters of the beam
        self.E = E
        self.I = I
        
        #The mass densit; currently not properly implemented
        self.mu = 1
        
        #Length of the beam and discretization parameter
        self.N = N
        self.L = L
        
        
        #The conditions used to initialize the state
        self.initial_conditions = initial_conditions
        
        #Physical parameters governing the dynamics of the beam
        self.parameters = parameters

        self.two_sided_support = two_sided_support
        
        
        #Generate the mesh
        self.nodes, self.elems, self.faces = get_mesh_1D(self.N, [0,self.L])
        
        
        #Define the initial object, and the one governing the dynamics
        if self.two_sided_support:
            self.initial_object = NumericalSolutionBothEnds(
                                                           M0 = self.initial_conditions["M0"],
                                                           ML = self.initial_conditions["ML"],
                                                           a0 = self.initial_conditions["a0"],
                                                           aL = self.initial_conditions["aL"],
                                                           q = self.initial_conditions["q"],
                                                           E = self.E, I = self.I, N = self.N, L = self.L
                                                           )
            
            self.dynamic_object = NumericalSolutionBothEnds(
                                                           M0 = self.parameters["M0"],
                                                           ML = self.parameters["ML"],
                                                           a0 = self.parameters["a0"],
                                                           aL = self.parameters["aL"],
                                                           q = self.parameters["q"],
                                                           E = self.E, I = self.I, N = self.N, L = self.L
                                                           )
            
        else:
            self.initial_object = NumericalSolutionCantilever(a = self.initial_conditions["a"],
                                                           b = self.initial_conditions["b"],
                                                           QL = self.initial_conditions["QL"],
                                                           ML = self.initial_conditions["ML"],
                                                           q = self.initial_conditions["q"],
                                                           E = self.E, I = self.I, N = self.N, L = self.L
                                                           )
            self.dynamic_object = NumericalSolutionCantilever(a = self.parameters["a"],
                                                           b = self.parameters["b"],
                                                           QL = self.parameters["QL"],
                                                           ML = self.parameters["ML"],
                                                           q = self.parameters["q"],
                                                           E = self.E, I = self.I, N = self.N, L = self.L
                                                           )
            
        #Get the initial state
        self.initial_state = self.initial_object.solve(raw_sol = True)
        
        
        #Get the necessary matrices
        self.S = self.dynamic_object.S_ext
        
        self.M = self.__get_extended_M(self.nodes, self.elems, self.mu, self.two_sided_support)
        
        self.RHS = self.dynamic_object.RHS_ext
        
        
        #Define the Newmark solver, using the previous matrices and states
        self.Newmark = Newmark(u1=self.initial_state, 
                               up1=np.zeros(np.shape(self.initial_state)), 
                               upp1=np.zeros(np.shape(self.initial_state)), 
                               h=0.1, beta=1/4, gamma=1/2,
                               Me=self.M, Se=self.S, 
                               f=self.RHS)
        
        
        
        
    def __get_extended_M(self, node_matrix, element_matrix, mu, two_sided_support=False):
        # Extend the mass matrix M
        # two_sided_support = True,  corresponds to a beam supported at both ends
        # two_sided_support = False, corresponds to a beam supported at the left

        n_p = np.shape(node_matrix)[0]

        M = self.__get_M_global(node_matrix, element_matrix, mu)
        # Rebuild the sparse mass matrix, but with 2 extra rows and 2 extra columns of zeros at the end
        indptr = np.hstack((M.indptr, M.indptr[-1], M.indptr[-1]))  # fix the pointers such that we can change the shape
        M_ext = csr_matrix((M.data, M.indices, indptr), shape=(2*n_p+2, 2*n_p+2))


        #make it possible to input mu (mass density) as a distribution, not only a constant
        return M_ext

    
    def __get_M_global(self, node_matrix, element_matrix, mu):
        """
        #IMPLEMENT THE MASS DENSITY MU
        """
        # Build matrix

        # number of nodes, elements and form functions
        n_p = np.shape(node_matrix)[0]
        n_e = np.shape(element_matrix)[0]
        n_phi = 4 

        e = 2*element_matrix  #expand indexing 

        ii = np.zeros((n_e, n_phi**2), dtype=np.int)   # row indices
        jj = np.zeros((n_e, n_phi**2), dtype=np.int)   # column indices
        ss = np.zeros((n_e, n_phi**2))                 # values 

        # Note there are two unknowns for every mesh point
        # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1
        # Hence all even indices (starting at 0) correspond to form functions 1 and 3, defining the values in the nodes x_i
        # All odd indices correspond to form functions 2 and 4, which define the derivative in x_i, but do not change the value in the nodes
        for k in np.arange(0, n_e):
            M_loc = self.__get_M_loc(node_matrix, element_matrix, k, mu)
            ii[k,:] = np.array([
                [e[k,0]  , e[k,0]  , e[k,0]  , e[k,0]],
                [e[k,0]+1, e[k,0]+1, e[k,0]+1, e[k,0]+1],
                [e[k,1]  , e[k,1]  , e[k,1]  , e[k,1]],
                [e[k,1]+1, e[k,1]+1, e[k,1]+1, e[k,1]+1]
                 ]).flatten()
            jj[k,:] = np.array([
                [e[k,0], e[k,0]+1, e[k,1], e[k,1]+1],
                [e[k,0], e[k,0]+1, e[k,1], e[k,1]+1],
                [e[k,0], e[k,0]+1, e[k,1], e[k,1]+1],
                [e[k,0], e[k,0]+1, e[k,1], e[k,1]+1]
                 ]).flatten()

            ss[k,:] = M_loc.flatten()

        # Bluild the stiffness matrix
        M = csr_matrix((ss.flatten(),(ii.flatten(),jj.flatten())), shape=(2*n_p, 2*n_p))  

        return M


    def __get_M_loc(self, node_matrix, element_matrix, element_nr, mu):
        T = self.__get_transformation(node_matrix, element_matrix, element_nr)  # transformation from reference element [0,1] to element [x_i, x_i+1]
        h = T(1) - T(0)

        if not callable(mu):

            M_ref = np.array([
                [156  ,  22*h   ,  54  , -13*h   ],
                [ 22*h,   4*h**2,  13*h,  -3*h**2],
                [ 54  ,  13*h   , 156  , -22*h   ],
                [-13*h,  -3*h**2, -22*h,   4*h**2]
                ])

            M_loc = (1/420) * h * mu * M_ref 
            return M_loc



    def __get_transformation(self, node_matrix, element_matrix, element_nr):
        x1 = node_matrix[element_matrix[element_nr, 0]]
        x2 = node_matrix[element_matrix[element_nr, 1]]

        dx = x2 - x1 # length of interval

        T = lambda z: x1 + dx*z   # transformation from reference element [0,1] to element [x_i, x_i+1]
        return T



    def __get_elements(self, x):
        n_p = len(x) # number of points
        n_e = n_p-1  # number of elements

        e1 = np.arange(0,n_p-1)
        e2 = np.arange(1,n_p)
        e = np.vstack((e1, e2)).T

        return n_e, n_p, e
    

    