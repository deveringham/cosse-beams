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
# Newmark.py
# Classes representing FEM solutions with time discretization using Newmark's method to the 1D beam problem.
#######


#######
# Dependencies
#######
from scipy.sparse.linalg import spsolve


class Newmark:

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
                        "upp": [self.upp1], "h": [], "t": []}
        
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
            
            if log_values:
                
                self.results["u"].append(u)
                self.results["up"].append(up)
                self.results["upp"].append(upp)
                self.results["h"].append(h)
                self.results["t"].append(t)
                
        return u, up, upp, t
    