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
# AnalyticalSolutions.py
# Classes representing analytical solutions to the 1D beam problem.
#######


#######
# Dependencies
#######
import numpy as np


#######
# AnalyticalSolution
#   Private base class describing generalized analytical solutions to the 1D 
#   beam problem with various types of boundary conditions.
#######
class _AnalyticalSolution:

    ###
    # Constants - Default problem parameters
    ###
    DEFAULT_a = 0
    DEFAULT_b = 0
    DEFAULT_QL = 0
    DEFAULT_M0 = 0
    DEFAULT_ML = 0
    DEFAULT_a0 = 0
    DEFAULT_aL = 0
    DEFAULT_case = 'arbitrary function'
    DEFAULT_c = -0.01
    DEFAULT_x0 = 0
    DEFAULT_E = 1
    DEFAULT_I = 1
    DEFAULT_L = 1
    DEFAULT_q = lambda x: x

    ###
    # Constructor
    ###
    def __init__(self, a=0, b=0, QL=0, M0=0, ML=0, a0=0, aL=0, 
        case='arbitrary function', c=-0.01, x0=0, E=1, I=1, L=5,
        q=lambda x: x):
    
        # Initialize member variables
        self.E, self.I = E, I
        self.L = L
        self.a, self.b = a, b
        self.QL, self.x0 = QL, x0
        self.M0, self.ML = M0, ML
        self.a0, self.aL = a0, aL
        self.q, self.case = q, case
        
        # Handle special cases of q
        if self.case == "arbitrary function":

            # For arbitrary q given as a lambda, use quadrature to integrate 
            import scipy.integrate as integrate
            self.q1 = lambda x: integrate.quad(self.q, 0, x)[0]
            self.q2 = lambda x: integrate.quad(self.q1, 0, x)[0]
            self.q3 = lambda x: integrate.quad(self.q2, 0, x)[0]
            self.q4 = lambda x: integrate.quad(self.q3, 0, x)[0]

        elif self.case == "constant":

            # For constant q, integrate analytically
            self.q1 = lambda x: c*x
            self.q2 = lambda x: c*np.power(x,2)/2
            self.q3 = lambda x: c*np.power(x,3)/6
            self.q4 = lambda x: c*np.power(x,4)/24

        elif self.case == "delta":

            # For q as a delta function, integrate analytically
            self.q1 = lambda x: 0 if x<x0 else 1
            self.q2 = lambda x: 0 if x<x0 else c*x
            self.q3 = lambda x: 0 if x<x0 else c*x*x/2
            self.q3 = lambda x: 0 if x<x0 else c*x*x*x/6
            self.q4 = lambda x: 0 if x<x0 else c*x*x*x*x/24
    
    ###
    # solve
    #   Class method which returns the analytical solution and its derivative
    #   as two lambda functions. Also saves them as class members.
    #
    # arguments: none
    # returns:
    #   w (lambda float -> float) : solution function
    #   wp (lambda float -> float) : solution derivative function
    ###
    def solve(self):
        coef = self.get_coefficients()

        w_first_term = lambda x : self.q4(x) / (self.E * self.I)
        w_polynomial = lambda x : coef[0] + coef[1]*x + \
                                coef[2]*x**2 + coef[3]*x**3
        self.w = lambda x: w_first_term(x) + w_polynomial(x)

        wp_first_term = lambda x : self.q3(x) / (self.E * self.I)
        wp_polynomial = lambda x : coef[1] + 2*coef[2]*x + 3*coef[3]*x**2
        self.wp = lambda x: wp_first_term(x) + wp_polynomial(x)

        return self.w, self.wp

#######
# AnalyticalSolutionCanteliver
#
#   Subclass of AnalyticalSolution describing solutions to the specific
#   problem of a 1D beam supported at one end.
#
#   In this case, M0, a0 and aL are not required.
#######
class AnalyticalSolutionCantilever(_AnalyticalSolution):
    
    def __init__(self, 
        a = _AnalyticalSolution.DEFAULT_a,
        b = _AnalyticalSolution.DEFAULT_b,
        QL = _AnalyticalSolution.DEFAULT_QL,
        ML = _AnalyticalSolution.DEFAULT_ML,
        x0 = _AnalyticalSolution.DEFAULT_x0,
        case = _AnalyticalSolution.DEFAULT_case,
        c = _AnalyticalSolution.DEFAULT_c,
        E = _AnalyticalSolution.DEFAULT_E,
        I = _AnalyticalSolution.DEFAULT_I,
        L = _AnalyticalSolution.DEFAULT_L,
        q = _AnalyticalSolution.DEFAULT_q):

        super().__init__(a=a, b=b, QL=QL, M0=0, ML=ML, a0=0, aL=0, x0=x0,
            case=case, c=c, E=E, I=I, L=L, q=q)

    def get_coefficients(self):
        
        EI = self.E * self.I

        c1 = self.a
        c2 = self.b
        c3 = (self.ML + self.QL*self.L + \
            self.q1(self.L)*self.L - self.q2(self.L))/(2*EI)
        c4 = -(self.QL + self.q1(self.L))/(6*EI)

        return [c1, c2, c3, c4]

#######
# AnalyticalSolutionBothEnds
#
#   Subclass of AnalyticalSolution describing solutions to the specific
#   problem of a 1D beam supported at both ends.
#
#   In this case, a, b and QL are not required.
#######
class AnalyticalSolutionBothEnds(_AnalyticalSolution):
    
    def __init__(self,
        a0 = _AnalyticalSolution.DEFAULT_a0,
        aL = _AnalyticalSolution.DEFAULT_aL,
        M0 = _AnalyticalSolution.DEFAULT_M0,
        ML = _AnalyticalSolution.DEFAULT_ML,
        x0 = _AnalyticalSolution.DEFAULT_x0,
        case = _AnalyticalSolution.DEFAULT_case,
        c = _AnalyticalSolution.DEFAULT_c,
        E = _AnalyticalSolution.DEFAULT_E,
        I = _AnalyticalSolution.DEFAULT_I,
        L = _AnalyticalSolution.DEFAULT_L,
        q = _AnalyticalSolution.DEFAULT_q):
        
        super().__init__(a=0, b=0, QL=0, M0=M0, ML=ML, a0=a0, aL=aL, x0=x0,
            case=case, c=c, E=E, I=I, L=L, q=q)
            
    def get_coefficients(self):
        
        EI = self.E * self.I

        c1 = self.a0
        c3 = -self.M0/(2*EI)
        c4 = (self.M0 + self.ML - self.q2(self.L))/(6*self.L*EI)
        c2 = (self.aL - self.q4(self.L)/EI - c1 - c3*self.L**2 - c4*self.L**3)/self.L
        
        return [c1, c2, c3, c4]