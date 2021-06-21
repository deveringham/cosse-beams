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
# 10.06.21
#
# Eigenvalues.py
# Classes representing transient FEM solutions using eigenvalue analysis.
#######


#######
# Dependencies
#######
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import inv, eigs
import numpy as np
from meshes import *
import warnings

# Turn off sparse matrix efficiency warnings
warnings.simplefilter('ignore', SparseEfficiencyWarning)

#######
# Eigenvalue
#   Private base class solving the following beam
#   problems with zero external forcess (i.e. q=0)
#   with eigenvalue/eigenvector analysis:
#
#   1. Cantilever with a (w(0))=b (w'(0))=ML=QL=0.
#   (clamped at the left end and free at the
#   right without bending moment or load)
#   2. Simply supported beam with
#       a=w(0)=w(L)=w''(0)=w''(L)=0
#   (no bending moment at end points)
#######


class Eigenvalues:

    ###
    # Constructor
    ###
    def __init__(self, w0, wp0, M, Me, Se):

        # Set initial values of w
        self.w0 = w0
        self.wp0 = wp0

        # Matrices and arrays needed for the solver
        self.M = M      # mass matrix
        self.Me = Me    # extended mass matrix
        self.Se = Se    # extended stiffness matrix

        self.A = inv(self.Se) @ self.Me    # eigenvalue operator

        # Matrix dimensions
        self.N = np.shape(self.M)[0]
        self.K = np.shape(self.Se)[0] - self.N

        self.eigvals, self.eigvecs = self.get_eigen()

        assert self.K == 2

    def get_eigen(self):
        # Calculates the eigenvalues and eigenvectors corresponding
        # to non-zero eigenvalues of A
        # Returns: arrays of non-zero eigenvalues and corresponding eigenvectors

        N = self.N
        K = self.K

        ### ----- Attempt with numpy ----- ###
        """A = self.A.toarray()
        eigvals, eigvecs = np.linalg.eig(A)
        idx = eigvals.argsort()[-(N-K):][::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]"""

        ### ----- Attempt with sparse (scipy) ----- #
        # Extract k=N-K eigenvalues of largest magnitude (=LM)
        # and corresponding eigenvectors of A 
        # The eigenvalue 0 has algebraic multiplicity 2*K. We do not find those
        # using this scipy.sparse.linalg.eigs routine
        A = self.A  # dimension = N+K
        eigvals, eigvecs = eigs(A, k=N-K, which='LM', tol=1e-16)

<<<<<<< Updated upstream
        #Make sure that all eigenvalues are real
=======
        # Make sure that all eigenvalues are real
>>>>>>> Stashed changes
        assert np.allclose(np.imag(eigvals), np.zeros(N-K))
        assert np.allclose(np.imag(eigvals), np.zeros(eigvals.shape))
        
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
<<<<<<< Updated upstream
        
=======

>>>>>>> Stashed changes
        return eigvals, eigvecs

    def solver(self, times, get_nodes=False):
        # Input: times = numpy array of time steps [0,T]
        # Output: solutions matrices w and mu (time t at column t)

        eigvals = self.eigvals
        eigvecs = self.eigvecs
        N = self.N
        K = self.K
        M = self.M
        T = np.shape(times)[0]

        # Make sure there exists N-K linearly independent
        # eigenvectors corresponding to positive eigenvalues

        assert isinstance(N-K, int)
        assert np.shape(eigvals)[0] == N-K

        # Initialize matrix to hold solution at time t in column t
        u = np.zeros((N+K, T))   # first N entries correspond to w, last K entries to mu
        w0 = self.w0    # initial values for w(0)
        wp0 = self.wp0  # initial values for w'(0) (derivative w.r.t time)
        assert np.shape(w0)[0] == N

        ### Initialize eigennode matrix
        w1 = eigvecs[:N, 0]     # first eigenvector
        # parameters
        omega1 = 1 / np.sqrt(eigvals[0])
        alpha1 = (w1.T @ M @ w0) / (w1.T @ M @ w1)
        beta1 = (w1.T @ M @ wp0) / (w1.T @ M @ w1)
        # initialize u as a list containing the matrices
        # of the different summands (vibration modes)
        # for all time steps
        u = [np.outer(eigvecs[:, 0], alpha1*np.cos(omega1*times) + (beta1/omega1)*np.sin(omega1*times))]

        ### Construct sum according to (ii) in Prop. 2 in script ev_method_numerical

        for k in range(N-K):
            wk = eigvecs[:N, k]
            assert np.linalg.norm(wk) != 0.0
            assert np.shape(wk)[0] == N

            # parameters
            omegak = 1/np.sqrt(eigvals[k])
            alphak = (wk.T @ M @ w0)/(wk.T @ M @ wk)
            betak = (wk.T @ M @ wp0)/(wk.T @ M @ wk)

            ###
            # use outer product to keep the form of a matrix
            u = np.append(u, [np.outer(eigvecs[:, k], alphak*np.cos(omegak*times)+(betak/omegak)*np.sin(omegak*times))],
                          axis=0)

        if get_nodes:
            return u

        w = np.sum(u, axis=0)[:N, :]
        mu = np.sum(u, axis=0)[:N, :]

        return w, mu
