from scipy import integrate
from scipy.sparse import csr_matrix
import numpy as np

def get_extended_q_vector(node_matrix, element_matrix, q, boundary_conditions, two_sided_support=False):
    # Extend the load vector q with boundary conditions to obtain the rhs of the system
    # **boundary_conditions is a dict that contains the boundary conditions for either case
    # two_sided_support = True, corresponds to a beam supported at both ends
    # possible keys in the dictionary are M0, ML, a0, aL 
    # two_sided_support = False, corresponds to a beam supported at the left end (x=0) only (default)
    # possible keys in the dictionary are QL, ML, a, b 

    q_vec = get_q_vector(node_matrix, element_matrix, q)
    q_vec_ext = np.concatenate((q_vec, [0,0]))


    if two_sided_support:
        #  Unpack boundary conditions
        M0 = boundary_conditions['M0']
        ML = boundary_conditions['ML']
        a0 = boundary_conditions['a0']
        aL = boundary_conditions['aL']
        # Insert them in the load vector at the right locations
        q_vec_ext[1]  -=  M0
        q_vec_ext[-3] +=  ML
        q_vec_ext[-2] =   a0
        q_vec_ext[-1] =  -aL
        
    else:
        #  Unpack boundary conditions
        QL = boundary_conditions['QL']
        ML = boundary_conditions['ML']
        a  = boundary_conditions['a']
        b  = boundary_conditions['b']
        # Insert them in the load vector at the right locations
        q_vec_ext[-4] += QL
        q_vec_ext[-3] += ML
        q_vec_ext[-2] =  a
        q_vec_ext[-1] =  b

    return q_vec_ext




def get_q_vector(node_matrix, element_matrix, q):
    # Build load vector
    
    # Number of nodes, elements and form functions
    n_p = np.shape(node_matrix)[0]
    n_e = np.shape(element_matrix)[0]
    
    # Note there are two unknowns for every mesh point
    # Therefore, we expand the indexing  i, i+1 --> 2*i, 2*i+1
    # Hence all even indices (starting at 0) correspond to form functions 1 and 3, defining the values in the nodes x_i
    # All odd indices correspond to form functions 2 and 4, which define the derivative in x_i, but do not change the value in the nodes
    e = 2*element_matrix  #expand indexing 

    q_vec = np.zeros(2*n_p)
 
    for k in np.arange(0, n_e):
        q_vec_loc = get_q_loc(node_matrix, element_matrix, k, q)
     
        # On each element, the contribution from phi_1 corresponds to the node xi on the left (w_i, i = e[k,0])
        # contribution from phi_2 to the derivative in the node xi on the left (w_i+1, , i = e[k,0])
        # the contribution from phi_3 corresponds to the node xi+1 on the right (w_i, i = e[k,1])
        # contribution from phi_2 to the derivative in the node xi+1 on the right (w_i+1, , i = e[k,1])
        q_vec[e[k,0]]   += q_vec_loc[0]         
        q_vec[e[k,0]+1] += q_vec_loc[1]
        q_vec[e[k,1]]   += q_vec_loc[2]
        q_vec[e[k,1]+1] += q_vec_loc[3]
    
    return q_vec
    




def get_q_loc(node_matrix, element_matrix, element_nr, q):
    # Compute the q vector contributions to one element
    # Use numerical quadrature if q is not a constant
    
    T = get_transformation(node_matrix, element_matrix, element_nr)  # transformation from reference element [0,1] to element [x_i, x_i+1]
    element_length = T(1) - T(0)
    
    # If q is a function, compute load vector with quadrature
    if callable(q):     
        # Form functions phi_1 bar to phi_4 bar 
        phi1 = lambda z: 1 - 3*z**2 + 2*z**3       # phi_1 bar 
        phi2 = lambda z: z * (z-1)**2
        phi3 = lambda z: 3*z**2 - 2*z**3
        phi4 = lambda z: z**2 * (z-1)
        phis = [phi1, phi2, phi3, phi4]
        
        # Define a local stiffness matrix for one element. Size 4 since we have 4 form functions
        q_vec_loc = np.empty(4)
        
        for i in range(4):
            # We integrate over a reference element [0,1]. Variable transformation defined by T(z).
            # Because of the transformation we multiply by the Jacobian determinant of the transformation which is element_length
            
            integrand = lambda z: phis[i](z) * q(T(z)) 
            q_vec_loc[i] = integrate.quadrature(integrand, 0, 1)[0] * element_length
        return q_vec_loc

    # If q is constant, we can compute the integrals analytically
    phi_integrals_exact = np.array([1/2, 1/12, 1/2, -1/12])
    q_vec_loc = q * element_length * phi_integrals_exact
    return q_vec_loc



def get_extended_S(node_matrix, element_matrix, E, I, two_sided_support=False):
    # Extend the stiffness matrix S
    # two_sided_support = True,  corresponds to a beam supported at both ends
    # two_sided_support = False, corresponds to a beam supported at the left
    # end (x=0) only and the right end is freee (default)
    
    n_p = np.shape(node_matrix)[0]

    
    S = get_S_global(node_matrix, element_matrix, E, I)
    # Rebuild the sparse stiffness matrix, but with 2 extra rows and 2 extra columns of zeros at the end
    indptr = np.hstack((S.indptr, S.indptr[-1], S.indptr[-1]))  # fix the pointers such that we can change the shape
    S_ext = csr_matrix((S.data, S.indices, indptr), shape=(2*n_p+2, 2*n_p+2))
    
    
    # Stacking e0, eL, d0, dL to S by putting +-1 at the right location
    S_ext[0, 2*n_p] = 1  # adds e0 to 2*n_p+1'th column 
    S_ext[2*n_p, 0] = 1  # adds e0.T to 2*n_p+1'th row
    
    if two_sided_support:
        S_ext[2*n_p-2, -1] = -1  # hstacks -eL to the right
        S_ext[-1, 2*n_p-2] = -1  # vstacks -eL to the bottom
        return S_ext
       
    S_ext[1, -1] = 1  # hstacks d0 to the right
    S_ext[-1, 1] = 1  # vstacks d0.T to the bottom
    return S_ext
    
    
    

def get_S_global(node_matrix, element_matrix, E, I):
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
        S_loc = get_S_loc(node_matrix, element_matrix, k, E, I)
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