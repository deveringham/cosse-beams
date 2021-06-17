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
        
        assert self.K == 2

    def get_eigen(self):
        # Calculates the eigenvalues and eigenvectors corresponding
        # to non-zero eigenvalues of A
        # Returns: arrays of non-zero eigenvalues and corresponding eigenvectors

        # Attempt with numpy
        # A = self.A.toarray()
        # eigvals, eigvecs = np.linalg.eig(A)

        # Attempt with sparse (scipy)
        A = self.A  # dimension = N+K
        N = self.N
        K = self.K
        
        # Extract k=N-K eigenvalues of largest magnitude (=LM)
        # and corresponding eigenvectors of A 
        # The eigenvalue 0 has algebraic multiplicity 2*K. We do not find those
        # using this scipy.sparse.linalg.eigs routine 
        eigvals, eigvecs = eigs(A, k=N-K, which='LM')

        
        #Make sure that all eigenvalues are real
        assert np.allclose(np.imag(eigvals), np.zeros(N-K))
        assert np.allclose(np.imag(eigvals), np.zeros(eigvals.shape))
        
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        
        print(np.sort(eigvals))  #sorting is for checking. Not needed mathematically

        return eigvals, eigvecs
    

    def solver(self, times):
        # Input: times = numpy array of time steps [0,T]
        # Output: solutions matrices w and mu (time t at column t)

        eigvals, eigvecs = self.get_eigen()
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

        # Construct sum according to (ii) in Prop. 2 in script ev_method_numerical
        for k in range(N-K):
            wk = eigvecs[:N, k]
            assert np.linalg.norm(wk) != 0.0
            assert np.shape(wk)[0] == N

            omegak = 1/np.sqrt(eigvals[k])
            alphak = (wk.T @ M @ w0)/(wk.T @ M @ wk)
            betak = (wk.T @ M @ wp0)/(wk.T @ M @ wk)
            # use outer product to keep the form of a matrix
            u = u + np.outer(eigvecs[:, k], alphak*np.cos(omegak*times) + (betak/omegak)*np.sin(omegak*times))

        w = u[:N, :]
        mu = u[N:, :]

        return w, mu
