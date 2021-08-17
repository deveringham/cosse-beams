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

    ###
    # get_eigen
    #   Calculates the eigenvalues and eigenvectors corresponding to the
    #   nonzero eigenvalues of A.
    #
    # arguments: none
    # returns:
    #   eigvals (float array) : list of eigenvalues
    #   eigvecs (2D float array) : list of corresponding eigenvectors
    ###
    def get_eigen(self):
        # Attempt with sparse (scipy)
        A = self.A  # dimension = N+K
        N = self.N
        K = self.K
        
        # Extract k=N-K eigenvalues of largest magnitude (=LM)
        # and corresponding eigenvectors of A 
        # The eigenvalue 0 has algebraic multiplicity 2*K. We do not find those
        # using this scipy.sparse.linalg.eigs routine 
        eigvals, eigvecs = eigs(A, k=N-K, which='LM')

        # Make sure that all eigenvalues are real
        assert np.allclose(np.imag(eigvals), np.zeros(N-K))
        assert np.allclose(np.imag(eigvals), np.zeros(eigvals.shape))
        
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        return eigvals, eigvecs

    ###
    # solve
    #   Returns solution matrices w and mu.
    #
    # arguments:
    #   times (float array) : array of time steps
    # returns:
    #   w (2D float array) : solution matrix
    #   mu (2D float array) : derivative solution matrix
    ###
    def solve(self, times):

        eigvals, eigvecs = self.get_eigen()
        N = self.N
        K = self.K
        M = self.M

        # Make sure there exists N-K linearly independent
        # eigenvectors corresponding to positive eigenvalues
        assert isinstance(N-K, int)
        assert np.shape(eigvals)[0] == N-K

        # Initialize matrix to hold solution at time t in column t
        w0 = self.w0    # initial values for w(0)
        wp0 = self.wp0  # initial values for w'(0) (derivative w.r.t time)
        assert np.shape(w0)[0] == N

        vib_modes_w = []
        vib_modes_mu = []

        # Construct sum with (ii) in Prop. 2 in script ev_method_numerical
        for k in range(N-K):
            wk = eigvecs[:N, k]
            assert np.linalg.norm(wk) != 0.0
            assert np.shape(wk)[0] == N

            omegak = 1/np.sqrt(eigvals[k])
            alphak = (wk.T @ M @ w0)/(wk.T @ M @ wk)
            betak = (wk.T @ M @ wp0)/(wk.T @ M @ wk)
            # use outer product to keep the form of a matrix

            # create vibration modes corresponding to the solution w
            # first N entries correspond to w, last K entries to mu
            vib_modes_w.append(np.outer(eigvecs[:, k], 
                alphak*np.cos(omegak*times)
                + (betak/omegak)*np.sin(omegak*times))[:N, :])

            # create vibration modes corresponding to the solution mu
            vib_modes_mu.append(np.outer(eigvecs[:, k], 
                alphak * np.cos(omegak * times)
                + (betak / omegak) * np.sin(omegak * times))[N:, :])

        # vib_modes is an array of matrices, where entry i corresponds to the 
        # ith vibration mode matrix, with every column representing the solution 
        # over the whole domain for a particular time step

        vib_modes_w = np.asarray(vib_modes_w)
        vib_modes_mu = np.asarray(vib_modes_mu)

        return vib_modes_w, vib_modes_mu
