from scipy import integrate
from scipy.sparse import csr_matrix
import numpy as np 

def get_S_global(node_matrix, element_matrix, E, I):
    # Build matrix
    
    # number of nodes, elements and form functions
    n_p = np.shape(node_matrix)[0]
    n_e = np.shape(element_matrix)[0]
    n_phi = 4 
    
    
    ii = np.zeros((n_e, n_phi**2), dtype=np.int)   # row indices
    jj = np.zeros((n_e, n_phi**2), dtype=np.int)   # column indices
    ss = np.zeros((n_e, n_phi**2))                 # values 
    
    # Note there are two unknowns for every mesh point
    # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1
    # Hence all even indices (starting at 0) correspond to form functions 1 and 3, defining the values in the nodes x_i
    # All odd indices correspond to form functions 2 and 4, which define the derivative in x_i, but do not change the value in the nodes
    for k in np.arange(0, n_e):
        S_loc = get_S_loc(node_matrix, element_matrix, k, E, I)
        e = 2*element_matrix  #expand indexing 
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
        
    # Bluild the stiffness matrix
    S = csr_matrix((ss.flatten(),(ii.flatten(),jj.flatten())), shape=(2*n_p, 2*n_p))  
    
    return S




def get_S_loc(node_matrix, element_matrix, element_nr, E, I):
    T = get_transformation(node_matrix, element_matrix, element_nr)  # transformation from reference element [0,1] to element [x_i, x_i+1]
    element_length = T(1) - T(0)
    
    
    # E and/or I is not constant and we should compute S by numerical quadrature
    if callable(E) or callable(I):
        
        # Create function returning the numerical value if only one of them is a constant
        if not callable(E): 
            E_val = E
            E = lambda x: E_val  
        if not callable(I):
            I_val = I
            I = lambda x: I_val

        # Second derivative of form functions phi_1 bar to phi_4 bar wrt z
        phi1_zz = lambda z: 12*z-6      # second derivative of phi_1 bar to z
        phi2_zz = lambda z: 6*z-4
        phi3_zz = lambda z: 6-12*z
        phi4_zz = lambda z: 6*z-2
        
        phis_zz = [phi1_zz, phi2_zz, phi3_zz, phi4_zz]
        
        # Define a local stiffness matrix for one element. Size 4x4 since we have 4 form functions
        S_loc = np.empty((4,4))
        
        for i in range(4):
            for j in range(i):
                # We integrate over a reference element [0,1]. Variable transformation defined by T(z).
                # Because of the transformation we multiply by the Jacobian determinant of the transformation which is element_length
                # Because of the transformation we replace derivativs wrt x by derivatives wrt z. Chain rule implies divisoin by element_length^2 per second derivative (2x)
                integrand = lambda z: phis_zz[i](z) * phis_zz[j](z) * E(T(z)) * I(T(z))
                S_loc[i,j] = integrate.quadrature(integrand, 0, 1)[0] / element_length**3
                # Matrix is symmetric:
                S_loc[j,i] =  S_loc[i,j] 
        return S_loc
   
    
   


    # Assuming E and I are constants 
    ## Matrix entries in S_ref are exact results of the integrals of the form functions over the reference element [0,1]
    ## Because of the transformation we multiply by the Jacobian determinant of the transformation which is element_length
    ## Because of the transformation we replace derivativs wrt x by derivatives wrt z. Chain rule implies divisoin by element_length^2 per second derivative (2x)
    S_ref = np.array([
        [ 12,  6, -12,  6],
        [  6,  4,  -6,  2],
        [-12, -6,  12, -6],
        [  6,  2,  -6,  4]
        ])
    
    S_loc = E * I / element_length**3 * S_ref
    return S_loc



def get_transformation(node_matrix, element_matrix, element_nr):
    x1 = node_matrix[element_matrix[element_nr, 0]]
    x2 = node_matrix[element_matrix[element_nr, 1]]
  
    dx = x2 - x1 # length of interval
    
    T = lambda z: x1 + dx*z   # transformation from reference element [0,1] to element [x_i, x_i+1]
    return T



def get_elements(x):
    n_p = len(x) # number of points
    n_e = n_p-1  # number of elements
    
    e1 = np.arange(0,n_p-1)
    e2 = np.arange(1,n_p)
    e = np.vstack((e1, e2)).T

    return n_e, n_p, e