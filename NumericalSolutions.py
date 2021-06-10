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
# Carsten van de Kamp, Sergi Andreu,
# Sebastian Myrbäck, Dylan Everingham
#   
# 09.06.21
#######


#######
# Dependencies
#######
from scipy import integrate
from scipy.sparse import (csr_matrix, SparseEfficiencyWarning)
from scipy.sparse.linalg import spsolve
import numpy as np
from meshes import *
import warnings

# Turn off sparse matrix efficiency warnings
warnings.simplefilter('ignore', SparseEfficiencyWarning)

#######
# Constants
#######
DEFAULT_a = 0
DEFAULT_b = 0
DEFAULT_QL = 0
DEFAULT_ML = 0
DEFAULT_M0 = 0
DEFAULT_a0 = 0
DEFAULT_aL = 0
DEFAULT_E = 1
DEFAULT_I = 1
DEFAULT_N = 25
DEFAULT_L = 1
DEFAULT_q = lambda x: x
DEFAULT_two_sided_support = False


class NumericalSolution:

    ###
    # Constructor
    ###
    def __init__(self, a, b, QL, ML, M0, a0, aL, E, I, N, L, q, two_sided_support):

        # Initialize member variables
        self.a, self.b = a, b
        self.QL, self.ML, self.M0 = QL, ML, M0
        self.a0, self.aL = a0, aL
        self.E, self.I = E, I
        self.N, self.L = N, L
        self.q = q
        self.two_sided_support = two_sided_support

        # Generate mesh
        self.nodes, self.elems, self.faces = get_mesh_1D(self.N, [0,self.L])

        # Get the extended S matrix
        self.S_ext = self.__get_extended_S()

        # Create the RHS
        self.RHS_ext = self.__get_extended_q_vector()

    def solve(self):
        # Solve system with scipy sparse solver
        sol = spsolve(self.S_ext, self.RHS_ext)
        sol = sol[:-2]

        # Recover w and w'
        self.w = sol[::2]
        self.wp = sol[1::2]

        return self.w, self.wp

    def __get_extended_q_vector(self):
        """ 
        Extend the load vector q with boundary conditions to obtain the rhs of the system
        **boundary_conditions is a dict that contains the boundary conditions for either case
        two_sided_support = True, corresponds to a beam supported at both ends
        possible keys in the dictionary are M0, ML, a0, aL 
        two_sided_support = False, corresponds to a beam supported at the left end (x=0) only (default)
        possible keys in the dictionary are QL, ML, a, b 
        """ 

        q_vec = self.__get_q_vector()
        q_vec_ext = np.concatenate((q_vec, [0,0]))

        # Insert boundary conditions in the load vector at the right locations
        if self.two_sided_support:
            q_vec_ext[1]  -=  self.M0
            q_vec_ext[-3] +=  self.ML
            q_vec_ext[-2] =   self.a0
            q_vec_ext[-1] =  -self.aL
            
        else:
            q_vec_ext[-4] += self.QL
            q_vec_ext[-3] += self.ML
            q_vec_ext[-2] =  self.a
            q_vec_ext[-1] =  self.b

        return q_vec_ext

    def __get_q_vector(self):
        """
        Build load vector
        """
        
        # Number of nodes, elements and form functions
        n_p = np.shape(self.nodes)[0]
        n_e = np.shape(self.elems)[0]
        
        # Note there are two unknowns for every mesh point
        # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1
        # Hence all even indices (starting at 0) correspond to form functions 1 and 3, defining the values in the nodes x_i
        # All odd indices correspond to form functions 2 and 4, which define the derivative in x_i, but do not change the value in the nodes
        e = 2 * self.elems  # expand indexing 

        q_vec = np.zeros(2*n_p)
     
        for k in np.arange(0, n_e):
            q_vec_loc = self.__get_q_loc(k)
         
            # On each element, the contribution from phi_1 corresponds to the node xi on the left (w_i, i = e[k,0])
            # contribution from phi_2 to the derivative in the node xi on the left (w_i+1, , i = e[k,0])
            # the contribution from phi_3 corresponds to the node xi+1 on the right (w_i, i = e[k,1])
            # contribution from phi_2 to the derivative in the node xi+1 on the right (w_i+1, , i = e[k,1])
            q_vec[e[k,0]]   += q_vec_loc[0]         
            q_vec[e[k,0]+1] += q_vec_loc[1]
            q_vec[e[k,1]]   += q_vec_loc[2]
            q_vec[e[k,1]+1] += q_vec_loc[3]

        return q_vec
        
    def __get_q_loc(self, el):
        """
        Compute the q vector contributions to one element
        Use numerical quadrature if q is not a constant
        """
        
        T = self.__get_transformation(el)  # transformation from reference element [0,1] to element [x_i, x_i+1]
        h = T(1) - T(0)  # length of the element
        
        # If q is a function, compute load vector with quadrature
        if callable(self.q):     
            # Form functions phi_1 bar to phi_4 bar 
            phi1 = lambda z: 1 - 3*z**2 + 2*z**3       # phi_1 bar 
            phi2 = lambda z: (z * (z-1)**2) * h
            phi3 = lambda z: 3*z**2 - 2*z**3
            phi4 = lambda z: (z**2 * (z-1)) * h
            phis = [phi1, phi2, phi3, phi4]
            
            # Define a local stiffness matrix for one element. Size 4 since we have 4 form functions
            q_vec_loc = np.empty(4)
            
            for i in range(4):
                # We integrate over a reference element [0,1]. Variable transformation defined by T(z).
                # Because of the transformation we multiply by the Jacobian determinant of the transformation,
                # which is the length of the element h
                integrand = lambda z: phis[i](z) * self.q(T(z)) 
                q_vec_loc[i] = integrate.quadrature(integrand, 0, 1)[0] * h

            return q_vec_loc

        # If q is constant, we can compute the integrals analytically
        phi_integrals_exact = np.array([1/2, 1/12*h, 1/2, -1/12*h])
        q_vec_loc = self.q * h * phi_integrals_exact

        return q_vec_loc

    def __get_extended_S(self):
        """
        Extend the stiffness matrix S
        two_sided_support = True,  corresponds to a beam supported at both ends
        two_sided_support = False, corresponds to a beam supported at the left
        end (x=0) only and the right end is freee (default)
        """
        
        n_p = np.shape(self.nodes)[0]

        # Convert to dense to avoid expensive element-by-element operations on csr matrices
        S = self.__get_S_global()

        # Rebuild the sparse stiffness matrix, but with 2 extra rows and 2 extra columns of zeros at the end
        indptr = np.hstack((S.indptr, S.indptr[-1], S.indptr[-1]))  # fix the pointers such that we can change the shape
        S_ext = csr_matrix((S.data, S.indices, indptr), shape=(2*n_p+2, 2*n_p+2))
        
        
        # Stacking e0, eL, d0, dL to S by putting +-1 at the right location
        S_ext[0, 2*n_p] = 1  # adds e0 to 2*n_p+1'th column 
        S_ext[2*n_p, 0] = 1  # adds e0.T to 2*n_p+1'th row
        
        if self.two_sided_support:
            S_ext[2*n_p-2, -1] = -1  # hstacks -eL to the right
            S_ext[-1, 2*n_p-2] = -1  # vstacks -eL to the bottom
            return S_ext
           
        S_ext[1, -1] = 1  # hstacks d0 to the right
        S_ext[-1, 1] = 1  # vstacks d0.T to the bottom

        # convert back to csr matrix and return
        return S_ext
        
    def __get_S_global(self):
        """
        Build matrix
        """
        
        # number of nodes, elements and form functions
        n_p = np.shape(self.nodes)[0]
        n_e = np.shape(self.elems)[0]
        n_phi = 4 
        
        e = 2 * self.elems  # expand indexing 
        
        ii = np.zeros((n_e, n_phi**2), dtype=np.int)   # row indices
        jj = np.zeros((n_e, n_phi**2), dtype=np.int)   # column indices
        ss = np.zeros((n_e, n_phi**2))                 # values 
        
        # Note there are two unknowns for every mesh point
        # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1
        # Hence all even indices (starting at 0) correspond to form functions 1 and 3, defining the values in the nodes x_i
        # All odd indices correspond to form functions 2 and 4, which define the derivative in x_i, but do not change the value in the nodes
        for k in np.arange(0, n_e):
            S_loc = self.__get_S_loc(k)
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
            
            ss[k,:] = S_loc.flatten()
            
        # Build the stiffness matrix
        S = csr_matrix((ss.flatten(),(ii.flatten(),jj.flatten())), shape=(2*n_p, 2*n_p))  
        
        return S

    def __get_S_loc(self, el):
        """
        Generate local stiffness matrix
        """ 
        T = self.__get_transformation(el)  # transformation from reference element [0,1] to element [x_i, x_i+1]
        h = T(1) - T(0)  # length of the element
        
        
        # E and/or I is not constant and we should compute S by numerical quadrature
        if callable(self.E) or callable(self.I):
            
            # Create function returning the numerical value if only one of them is a constant
            if not callable(self.E): 
                E_val = self.E
                E = lambda x: E_val  
            if not callable(self.I):
                I_val = self.I
                I = lambda x: I_val

            # Second derivative of form functions phi_1 bar to phi_4 bar wrt z
            phi1_zz = lambda z: 12*z-6      # second derivative of phi_1 bar to z
            phi2_zz = lambda z: (6*z-4) * h
            phi3_zz = lambda z: 6-12*z
            phi4_zz = lambda z: (6*z-2) * h
            
            phis_zz = [phi1_zz, phi2_zz, phi3_zz, phi4_zz]
            
            # Define a local stiffness matrix for one element. Size 4x4 since we have 4 form functions
            S_loc = np.empty((4,4))
            
            for i in range(4):
                for j in range(i):
                    # We integrate over a reference element [0,1]. Variable transformation defined by T(z).
                    # Because of the transformation we multiply by the Jacobian determinant of the transformation which is the length of the element h
                    # Because of the transformation we replace derivativs wrt x by derivatives wrt z. Chain rule implies divisoin by h^2 per second derivative (2x)
                    integrand = lambda z: phis_zz[i](z) * phis_zz[j](z) * self.E(T(z)) * self.I(T(z))
                    S_loc[i,j] = integrate.quadrature(integrand, 0, 1)[0] / h**3

                    # Matrix is symmetric:
                    S_loc[j,i] =  S_loc[i,j]

            return S_loc
       
        
        # Assuming E and I are constants 
        ## Matrix entries in S_ref are exact results of the integrals of the form functions over the reference element [0,1]
        ## Because of the transformation we multiply by the Jacobian determinant of the transformation which is the length of the element h
        ## Because of the transformation we replace derivativs wrt x by derivatives wrt z. Chain rule implies divisoin by h^2 per second derivative (2x)
        S_ref = np.array([
            [ 12  ,  6*h   , -12  ,  6*h   ],
            [  6*h,  4*h**2,  -6*h,  2*h**2],
            [-12  , -6*h   ,  12  , -6*h   ],
            [  6*h,  2*h**2,  -6*h,  4*h**2]
            ])
        
        S_loc = self.E * self.I * S_ref / h**3
        return S_loc

    def __get_transformation(self, el):
        """
        Get transformation function for the transformation from the reference element [0,1] to an arbitrary element [x_i, x_i+1]
        """
        
        x1 = self.nodes[self.elems[el, 0]]
        x2 = self.nodes[self.elems[el, 1]]
      
        h = x2 - x1 # length of element
        
        T = lambda z: x1 + h*z   # transformation from reference element [0,1] to element [x_i, x_i+1]
        
        return T


#######
# Subclasses
#######
class NumericalSolutionCantiliver(NumericalSolution):
    def __init__(self, 
        a = DEFAULT_a,
        b = DEFAULT_b, 
        QL = DEFAULT_QL,
        ML = DEFAULT_ML,
        E = DEFAULT_E, 
        I = DEFAULT_I,
        N = DEFAULT_N, 
        L = DEFAULT_L, 
        q = DEFAULT_q):

        super().__init__(a=a, b=b, QL=QL, ML=ML, M0=0, a0=0, aL=0,
            E=E, I=I, N=N, L=L, q=q, two_sided_support=False)


class NumericalSolutionBothEnds(NumericalSolution):
    def __init__(self,
        ML = DEFAULT_ML,
        M0 = DEFAULT_M0,
        a0 = DEFAULT_a0,
        aL = DEFAULT_aL,
        E = DEFAULT_E, 
        I = DEFAULT_I,
        N = DEFAULT_N, 
        L = DEFAULT_L, 
        q = DEFAULT_q):

        super().__init__(a=0, b=0, QL=0, ML=ML, M0=M0, a0=a0, aL=aL,
            E=E, I=I, N=N, L=L, q=q, two_sided_support=True)