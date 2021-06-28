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
# 27.06.21
#
# DynamicSolutions.py
# Classes representing FEM solutions with time discretization
# using Newmark's method for the 1D beam problem.
#######

# TODO:
#	- Make it possible to input mu (mass density) as a distribution
#	(arbitrary function), not just a constant


#######
# Dependencies
#######
from Constants import *
from NumericalSolutions import *
from scipy import integrate
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse.linalg import spsolve
from meshes import *


#######
# _Newark
#   Private base describing the dynamic evolution of the generalized
#	time-dependent 1D beam problem, using the Newmark method.
#######
class _Newmark:

	###
    # Constructor
    ###
    def __init__(self, 
    	u1 = DEFAULT_u1,
    	up1 = DEFAULT_up1,
    	upp1 = DEFAULT_upp1,
    	h = DEFAULT_h,
    	beta = DEFAULT_beta,
    	gamma = DEFAULT_gamma,
    	Me = DEFAULT_Me,
        Se = DEFAULT_Se,
        f = DEFAULT_f):

        # Initialize member variables
        self.u1, self.up1, self.upp1 = u1, up1, upp1

        # Set timestep
        self.h = h

        #self.uniform = True
        #if not isinstance(self.h, float):
        #    self.uniform = False
        
        # Set numerical parameters
        assert beta > 0 and beta <=1/2
        assert gamma > 0 and gamma <=1
        self.beta, self.gamma = beta, gamma
        
        # Dictionary of arrays containing all the intermediate values
        self.results = {"u": [self.u1], "up": [self.up1],
                        "upp": [self.upp1], "h": [0], "t": [0],
                       "w":[self.u1[:-2][::2]], "wp":[self.u1[:-2][1::2]]}
        
        # Matrices and arrays needed for the solver
        self.Me, self.Se, self.f = Me, Se, f
        
    # The equation governing the dynamics is:
    # Me(x | mu)¨ + Se(x | mu) - (f | a) = 0
    # we are using u = (x | mu)

    def __get_ustars(self, u, up, upp):
        ustar = u + self.h * up + (0.5 - self.beta) * upp * self.h**2
        ustarp = up + (1 - self.gamma) * upp * self.h

        return ustar, ustarp

	    
    def __solve(self, ustar):

        # All matrices and vectors are given as numpy arrays
        rhs = self.f - self.Se @ ustar
        lhs = self.Me + self.beta * self.h**2 * self.Se
        
        # Solve and return result
        upp = spsolve(lhs, rhs)
        return upp
    
    
    def __get_next(self, ustar, upstar, upp):

        unext = ustar + self.beta * upp * self.h**2
        upnext = upstar + self.gamma * upp * self.h
        return unext, upnext
        
    
    def run(self, n, log_values=True):
        
        #assert self.uniform
        
        # Initialize the values at the initial b.c.
        u = self.u1
        up = self.up1
        upp = self.upp1
        
        # For each timestep...
        t = 0
        for i in range(n):
            
            t += self.h       
            ustar, upstar = self.__get_ustars(u, up, upp)
            
            # NOTE THAT THE SOLVER DOES NOT NEED ustarp

            upp = self.__solve(ustar)
            u, up = self.__get_next(ustar, upstar, upp)
            
            # In this case, the derivative is wrt the space variable
            w = u[:-2][::2]
            wp = u[:-2][1::2] 
            
            # Store intermediate values
            if log_values:
                self.results["u"].append(u)
                self.results["up"].append(up)
                self.results["upp"].append(upp)
                self.results["h"].append(self.h)
                self.results["t"].append(t)
                self.results["w"].append(w)
                self.results["wp"].append(wp)
                
        return u, up, upp, t


#######
# DynamicSolution
#   Private base class describing generalized FEM solutions to the
#	time-dependent 1D beam problem with various types of boundary conditions.
#######
class _DynamicSolution:

	###
	# Constructor
	###
	def __init__(self, E, I, N, L, h, initial_conditions, parameters, 
		two_sided_support):

		# Fixed parameters of the beam
		self.E, self.I = E, I

		# The mass density; currently not properly implemented
		self.mu = 1

		# Length of the beam and discretization parameter
		self.N, self.L = N, L

		# Timestep size
		self.h = h

		# The conditions used to initialize the state
		self.initial_conditions = initial_conditions

		# Physical parameters governing the dynamics of the beam
		self.parameters = parameters
		self.two_sided_support = two_sided_support

		# Generate the mesh
		self.nodes, self.elems, self.faces = get_mesh_1D(self.N, [0,self.L])

		# Numerical solution object representing initial state,
		# and numerical solution object representing dynamics
		if (two_sided_support):
			self.initial_object = NumericalSolutionBothEnds(
				M0=self.initial_conditions['M0'],
				ML=self.initial_conditions['ML'],
				a0=self.initial_conditions['a0'],
				aL=self.initial_conditions['aL'],
				q=self.initial_conditions['q'],
				E=self.E, I=self.I, N=self.N, L=self.L)

			self.dynamic_object = NumericalSolutionBothEnds(
				M0=self.parameters['M0'],
				ML=self.parameters['ML'],
				a0=self.parameters['a0'],
				aL=self.parameters['aL'],
				q=self.parameters['q'],
				E=self.E, I=self.I, N=self.N, L=self.L)

		else:
			self.initial_object = NumericalSolutionCantilever(
				a=self.initial_conditions['a'],
				b=self.initial_conditions['b'],
				QL=self.initial_conitions['QL'],
				ML=self.initial_conditions['ML'],
				q=self.initial_conditions['q'],
				E=self.E, I=self.I, N=self.N, L=self.L)

			self.dynamic_object = NumericalSolutionCantilever(
				a=self.parameters['a'],
				b=self.parameters['b'],
				QL=self.parameters['QL'],
				ML=self.parameters['ML'],
				q=self.parameters['q'],
				E=self.E, I=self.I, N=self.N, L=self.L)

		# Get the initial state
		self.initial_state = self.initial_object.solve(raw_sol=True)

		# Get the necessary matrices
		self.S_ext = self.dynamic_object.S_ext
		self.M_ext = self.__get_extended_M()
		self.RHS_ext = self.dynamic_object.RHS_ext

		# Define the Newmark solver, using the previous matrices and states
		self.Newmark = _Newmark(u1=self.initial_state, 
			up1=np.zeros(np.shape(self.initial_state)), 
			upp1=np.zeros(np.shape(self.initial_state)), 
			h=self.h, beta=1/4, gamma=1/2,
			Me=self.M_ext, Se=self.S_ext, 
			f=self.RHS_ext)
    
	###
	# run
	#	Runs time exolution for n steps, and returns all results at each step.
	#
	# arguments:
	#	n (int) : number of timesteps
	# returns:
	#	results : dictionary containing intermediate results
	###
	def run(self, n):

		# Run time evolution
		self.Newmark.run(n, log_values=True)

		# Return all intermediate results
		return self.Newmark.results

	###
	# get_extended_M
	#   Private class method to extend the mass matrix M.
	#
	# arguments: none
	# returns:
	#   M_ext (2Dlist of float) : extended mass matrix
	###
	def __get_extended_M(self):

		# Number of nodes
		n_p = np.shape(self.nodes)[0]

		M = self.__get_M_global()

		# Rebuild the sparse stiffness matrix, but with 2 extra rows and
		# 2 extra columns of zeros at the end
		indptr = np.hstack((M.indptr, M.indptr[-1], M.indptr[-1]))
		M_ext = csr_matrix((M.data, M.indices, indptr),
			shape=(2*n_p+2, 2*n_p+2))

		return M_ext

	###
	# get_M_global
	#   Private class method to build the mass matrix.
	#
	# arguments: none
	# returns:
	#   M (2Dlist of float) : mass matrix
	###
	def __get_M_global(self):

	    # Number of nodes, elements and form functions
	    n_p = np.shape(self.nodes)[0]
	    n_e = np.shape(self.elems)[0]
	    n_phi = 4

		# Note there are two unknowns for every mesh point (i.e. w and w')
	    # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1 .
	    #
	    # Hence all even indices (starting at 0) correspond to 
	    # form functions 1 and 3, defining the values in the nodes x_i, and
	    # all odd indices correspond to form functions 2 and 4, which define
	    # the derivative in x_i, but do not change the value in the nodes.
	    e = 2 * self.elems

	    ii = np.zeros((n_e, n_phi**2), dtype=np.int)   # Row indices
	    jj = np.zeros((n_e, n_phi**2), dtype=np.int)   # Column indices
	    ss = np.zeros((n_e, n_phi**2))                 # Values

		# For each element...
	    for k in np.arange(0, n_e):
	        M_loc = self.__get_M_loc(k)
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

	    # Build the mass matrix
	    M = csr_matrix((ss.flatten(),(ii.flatten(),jj.flatten())),
	   		shape=(2*n_p, 2*n_p))
	    return M

	###
	# get_M_loc
	#   Private class method to compute mass matrix contribution from
	#   one element. IF mu is not given as a function, we can compute it
	#	analytically. the opposite case (mu as an arbitrary function) not
	#	yet implemented.
	#
	# arguments:
	#   el (int) : element index into elements matrix
	# returns:
	#   M_loc (2Dlist of float) : local contribution to mass matrix
	###
	def __get_M_loc(self, el):

		# Get transformation from reference element [0,1] to 
	    # local element [x_i, x_i+1]
	    T = self.__get_transformation(el)
	    h = T(1) - T(0) # Length of the element

	    # Constant mu case
	    if not callable(self.mu):
	        M_ref = np.array([
	            [156  ,  22*h   ,  54  , -13*h   ],
	            [ 22*h,   4*h**2,  13*h,  -3*h**2],
	            [ 54  ,  13*h   , 156  , -22*h   ],
	            [-13*h,  -3*h**2, -22*h,   4*h**2]
	            ])
	        M_loc = (1/420) * h * self.mu * M_ref 
	    
	    # TODO: implement other case

	    return M_loc

	###
	# get_transformation
	#   Private class method to get transformation function for the
	#   transformation from the reference element to an arbitrary element.
	#
	# arguments:
	#   el (int) : element index into elements matrix
	# returns:
	#   T (lambda (list of float) -> (list of float)) : transformation function
	###
	def __get_transformation(self, el):
	    # Get the two nodes comprising this element
	    x1 = self.nodes[self.elems[el, 0]]
	    x2 = self.nodes[self.elems[el, 1]]
	  	
	    h = x2 - x1 # Length of element
	    
	    # Construct transformation function
	    T = lambda z: x1 + h*z
	    return T


#######
# DynamicSolutionCanteliver
#
#   Subclass of DynamicSolution describing FEM solutions to the specific
#   time-dependent problem of a 1D beam supported at one end.
#
#   In this case, M0, a0 and aL are not required, and two_sided_support
#   takes the value False.
#######
class DynamicSolutionCantilever(_DynamicSolution):
    def __init__(self,
    	E = DEFAULT_E,
    	I = DEFAULT_I,
    	N = DEFAULT_N,
    	L = DEFAULT_L,
    	h = DEFAULT_h,
    	initial_conditions = { \
    		"a":  DEFAULT_a,
    		"b":  DEFAULT_b,
    		"QL": DEFAULT_QL,
    		"ML": DEFAULT_ML,
    		"q":  DEFAULT_q },
    	parameters = { \
    	    "a":  DEFAULT_a,
    		"b":  DEFAULT_b,
    		"QL": DEFAULT_QL,
    		"ML": DEFAULT_ML,
    		"q":  DEFAULT_q } ):

        super().__init__(E=E, I=I, N=N, L=L, h=h,
        	initial_conditions=initial_conditions,
        	parameters=parameters,
        	two_sided_support=False)


#######
# DynamicSolutionBothEnds
#
#   Subclass of DynamicSolution describing FEM solutions to the specific
#   time-dependent problem of a 1D beam supported at both ends.
#
#   In this case, a, b and QL are not required, and two_sided_support
#   takes the value True.
#######
class DynamicSolutionBothEnds(_DynamicSolution):
    def __init__(self,
    	E = DEFAULT_E,
    	I = DEFAULT_I,
    	N = DEFAULT_N,
    	L = DEFAULT_L,
    	h = DEFAULT_h,
        initial_conditions = { \
    		"M0": DEFAULT_M0,
    		"ML": DEFAULT_ML,
    		"a0": DEFAULT_a0,
    		"aL": DEFAULT_aL,
    		"q":  DEFAULT_q },
    	parameters = { \
    		"M0": DEFAULT_M0,
    		"ML": DEFAULT_ML,
    		"a0": DEFAULT_a0,
    		"aL": DEFAULT_aL,
    		"q":  DEFAULT_q } ):

        super().__init__(E=E, I=I, N=N, L=L, h=h,
        	initial_conditions=initial_conditions,
        	parameters=parameters,
        	two_sided_support=True)