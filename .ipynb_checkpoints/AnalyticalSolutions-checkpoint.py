import numpy as np


class Cantilever:
    
    def __init__(self, a, b, QL, ML, x0, case, c=-0.01, E=1, I=1, L=5, q=lambda x: x):
        
        self.E = E
        self.I = I
        
        self.L = L
        
        self.a = a
        self.b = b
        self.QL = QL
        self.ML = ML
        self.x0 = x0
        
        self.c = c
        self.q = q
        
        self.case = case
        
        if self.case == "arbitrary function":
            
            import scipy.integrate as integrate

            self.q1 = lambda x: integrate.quad(self.q, 0, x)[0]
            self.q2 = lambda x: integrate.quad(self.q1, 0, x)[0]
            self.q3 = lambda x: integrate.quad(self.q2, 0, x)[0]
            self.q4 = lambda x: integrate.quad(self.q3, 0, x)[0]

        if self.case == "constant":
            self.q1 = lambda x: c*x
            self.q2 = lambda x: c*np.power(x,2)/2
            self.q3 = lambda x: c*np.power(x,3)/6
            self.q4 = lambda x: c*np.power(x,4)/24

        if self.case == "delta":
            self.q1 = lambda x: 0 if x<x0 else 1
            self.q2 = lambda x: 0 if x<x0 else c*x
            self.q3 = lambda x: 0 if x<x0 else c*x*x/2
            self.q3 = lambda x: 0 if x<x0 else c*x*x*x/6


        
    def get_coefficients(self):
        
        EI = self.E * self.I
        
        a1 = self.a
        a2 = self.b
        
        a3 = (self.ML + self.QL*self.L + self.q1(self.L)*self.L - self.q2(self.L))/(2*EI)

        a4 = -(self.QL + self.q1(self.L))/(6*EI)
        
        return [a1,a2,a3,a4]
    
    
    def get_solution_list(self, x):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.q4(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        first_term = np.vectorize(first_term)
        polynomial = np.vectorize(polynomial)

        return first_term(x) + polynomial(x)
    
    def get_solution(self):
        
        coef = self.get_coefficients()
        
        primitives_4 = self.q4
        
        first_term = lambda x : primitives_4(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        
        sol = lambda x: first_term(x) + polynomial(x)

        return sol
    
    def get_solution_prime(self):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.q3(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[1] + 2*coef[2]*x + 3*coef[3]*x**2
        
        
        sol = lambda x: first_term(x) + polynomial(x)

        return sol
    
    
class BothEnds:
    
    def __init__(self, A1, B1, A2, B2, x0, case, c=-0.01, E=1, I=1, L=5, q= lambda x: x):
        
        self.E = E
        self.I = I
        
        self.L = L
        
        self.A1 = A1
        self.B1 = B1
        self.A2 = A2
        self.B2 = B2
        self.x0 = x0
        
        self.c = c
        self.q = q
        
        
        self.case = case
        
        if self.case == "arbitrary function":
            
            import scipy.integrate as integrate

            self.q1 = lambda x: integrate.quad(self.q, 0, x)[0]
            self.q2 = lambda x: integrate.quad(self.q1, 0, x)[0]
            self.q3 = lambda x: integrate.quad(self.q2, 0, x)[0]
            self.q4 = lambda x: integrate.quad(self.q3, 0, x)[0]

        if self.case == "constant":
            self.q1 = lambda x: c*x
            self.q2 = lambda x: c*np.power(x,2)/2
            self.q3 = lambda x: c*np.power(x,3)/6
            self.q4 = lambda x: c*np.power(x,4)/24

        if self.case == "delta":
            self.q1 = lambda x: 0 if x<x0 else 1
            self.q2 = lambda x: 0 if x<x0 else c*x
            self.q3 = lambda x: 0 if x<x0 else c*x*x/2
            self.q3 = lambda x: 0 if x<x0 else c*x*x*x/6
            
        
    def get_coefficients(self):
        
        EI = self.E * self.I
        
        a1 = self.A1
        a2 = self.B1
        
        
        phi1 = self.q4(self.L)/EI
        phi2 = self.q3(self.L)/EI
        
        
        a4 = 2*(self.A1 - self.A2) + self.L*(self.B1 + self.B2) - self.L*phi2 + 2*phi1
        a4 = a4 / (self.L ** 3)
        
        a3 = (-3/2) * a4 * self.L **3 + (self.L/2)*(self.B2 - self.B1) - self.L*phi2/2
        
        a3 = a3/(self.L*self.L)
        
        return [a1, a2, a3, a4]
    
    
    def get_solution_list(self, x):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.q4(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        first_term = np.vectorize(first_term)
        polynomial = np.vectorize(polynomial)

        return first_term(x) + polynomial(x)
    
    def get_solution(self):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.q4(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        sol = lambda x: first_term(x) + polynomial(x)
        
        return sol
        
        
    def get_solution_prime(self):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.q3(x) / (self.E * self.I)
        
        polynomial = lambda x : coef[1] + 2*coef[2]*x + 3*coef[3]*x**2
        
        sol = lambda x: first_term(x) + polynomial(x)

        return sol