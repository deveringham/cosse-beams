import numpy as np

class Cantilever:
    
    def __init__(self, a, b, QL, ML, x0, case, c=-0.01, E=1, I=1, L=5):
        
        self.E = E
        self.I = I
        
        self.L = L
        
        self.a = a
        self.b = b
        self.QL = QL
        self.ML = ML
        self.x0 = x0
        
        self.case = case
        
        self.c = c
        
    def get_q_primitives(self, x, n):
        
        c = self.c
        
        if self.case == "constant":
            if n == 1:
                return c*x
            if n == 2:
                return c*np.power(x,2)/2
            if n == 3:
                return c*np.power(x,3)/6
            if n == 4:
                return c*np.power(x,4)/24
            
        if self.case == "delta":
            
            if x <= self.x0:
                return 0
            else:
                if n == 1:
                    return c
                if n == 2:
                    return c*(x-self.x0)
                if n == 3:
                    return c*(0.5*(x**2 + self.x0**2) - x*self.x0)  
                if n == 4:
                    return c*((1/6)*(x**3 - self.x0**3) - 0.5*(self.x0*self.x0*x - self.x0*x*x))
        
    def get_coefficients(self):
        
        EI = self.E * self.I
        
        a1 = self.a
        a2 = self.b
        
        num1 = self.ML - self.get_q_primitives(self.L, 2)
        den1 = 2*EI

        num2 = self.L * ( self.QL - self.get_q_primitives(self.L, 1) )
        den2 = EI
        
        a3 = num1/den1 + num2/den2

        num = self.QL - self.get_q_primitives(self.L, 1)
        den = 6*EI
        
        a4 = num/den
        
        return [a1,a2,a3,a4]
    
    
    def get_solution(self, x):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.get_q_primitives(x, 4) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        first_term = np.vectorize(first_term)
        polynomial = np.vectorize(polynomial)

        return first_term(x) + polynomial(x)
    
    
class BothEnds:
    
    def __init__(self, A1, B1, A2, B2, x0, case, c=-0.01, E=1, I=1, L=5):
        
        self.E = E
        self.I = I
        
        self.L = L
        
        self.A1 = A1
        self.B1 = B1
        self.A2 = A2
        self.B2 = B2
        self.x0 = x0
        
        self.case = case
        
        self.c = c
        
    def get_q_primitives(self, x, n):
        
        c = self.c
        
        if self.case == "constant":
            if n == 1:
                return c*x
            if n == 2:
                return c*np.power(x,2)/2
            if n == 3:
                return c*np.power(x,3)/6
            if n == 4:
                return c*np.power(x,4)/24
            
        if self.case == "delta":
            
            if x <= self.x0:
                return 0
            else:
                if n == 1:
                    return c
                if n == 2:
                    return c*(x-self.x0)
                if n == 3:
                    return c*(0.5*(x**2 + self.x0**2) - x*self.x0)  
                if n == 4:
                    return c*((1/6)*(x**3 - self.x0**3) - 0.5*(self.x0*self.x0*x - self.x0*x*x))
        
    def get_coefficients(self):
        
        EI = self.E * self.I
        
        a1 = self.A1
        a2 = self.B1
        
        term1 = (1/self.L**2)*((self.B2 + self.B1)/2 + ((self.A1 - self.A2)/self.L))
        term2 = (1/(EI*self.L**2))*( (self.get_q_primitives(self.L, 4)/self.L) - (self.get_q_primitives(self.L, 3)/2) )
        
        a4 = term1+term2
        
        a3 = -(3*self.L/2)*a4 + (self.B2 - self.B1)/(2*self.L) - (self.get_q_primitives(self.L, 3)/(2*self.L*EI))
        
        return [a1,a2,a3,a4]
    
    
    def get_solution(self, x):
        
        coef = self.get_coefficients()
        
        first_term = lambda x : self.get_q_primitives(x, 4) / (self.E * self.I)
        
        polynomial = lambda x : coef[0] + coef[1]*x + coef[2]*x**2 + coef[3]*x**3
        
        first_term = np.vectorize(first_term)
        polynomial = np.vectorize(polynomial)

        return first_term(x) + polynomial(x)