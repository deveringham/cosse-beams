from scipy import integrate
from scipy.sparse import csr_matrix
import numpy as np




def get_extended_M(node_matrix, element_matrix, mu, two_sided_support=False):
    # Extend the mass matrix M
    # two_sided_support = True,  corresponds to a beam supported at both ends
    # two_sided_support = False, corresponds to a beam supported at the left
    
    n_p = np.shape(node_matrix)[0]

    M = get_M_global(node_matrix, element_matrix, mu)
    # Rebuild the sparse mass matrix, but with 2 extra rows and 2 extra columns of zeros at the end
    indptr = np.hstack((M.indptr, M.indptr[-1], M.indptr[-1]))  # fix the pointers such that we can change the shape
    M_ext = csr_matrix((M.data, M.indices, indptr), shape=(2*n_p+2, 2*n_p+2))
    
    
    #WHICH CHANGES SHOULD WE MAKE TO THE EXTENDED MASS MATRIX, DEPENDING ON THE B.C.?
    #The following are probably wrong: 
    
    M_ext[0, 2*n_p] = 1  # adds e0 to 2*n_p+1'th column 
    M_ext[2*n_p, 0] = 1  # adds e0.T to 2*n_p+1'th row
    
    if two_sided_support:
        M_ext[2*n_p-2, -1] = -1  # hstacks -eL to the right
        M_ext[-1, 2*n_p-2] = -1  # vstacks -eL to the bottom
        return M_ext
       
    M_ext[1, -1] = 1  # hstacks d0 to the right
    M_ext[-1, 1] = 1  # vstacks d0.T to the bottom
    
    #make it possible to input mu (mass density) as a distribution, not only a constant
    return M_ext
    
    
    

def get_M_global(node_matrix, element_matrix, mu):
    
    #IMPLEMENT THE MASS DENSITY MU
    
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
        M_loc = get_M_loc(node_matrix, element_matrix, k)
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




def get_M_loc(node_matrix, element_matrix, element_nr):
    T = get_transformation(node_matrix, element_matrix, element_nr)  # transformation from reference element [0,1] to element [x_i, x_i+1]
    element_length = T(1) - T(0)
    
    
    M_ref = np.array([
        [156,  22,  54, -13],
        [ 22,  14,  13,  -3],
        [ 54,  13,  16, -22],
        [-13,  -3, -22,   4]
        ])
    
    M_loc = (1 / element_length) * M_ref
    return M_loc



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