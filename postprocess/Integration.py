"""
Integration object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np

class Integration():

    def __init__(self, physics):
        """
        Args:
        physics (Physics object) - takes in a physics object to be integrated
        """

        self.physics = physics

    def rk4_step(t_n, u_n, dudt, dt): 
        """
        Computes the next timestep of u_n given the finite difference function du/dt
        with a 4-stage, 4th order accurate Runge-Kutta method. 
        
        Parameters
        ----------
        t_n : float
            time for time step n
        u_n : array-like
            condition at time step n
        dudt : function 
            function du/dt(t, u)
        dt : float
            time step
        
        Returns u_(n+1)
        """    
        k1 = dt * dudt(t_n, u_n)
        k2 = dt * dudt(t_n + dt/2, u_n + k1/2)
        k3 = dt * dudt(t_n + dt/2, u_n + k2/2)
        k4 = dt * dudt(t_n + dt, u_n + k3)

        u_n1 = u_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        return u_n1


    def EF_step(t_n, u_n, dudt, dt): 
        """
        Forward Euler stepping scheme
        """
        u_n1 = u_n + dt * dudt(t_n, u_n)
        return u_n1


    def integrate(self, u0=None, T=[0, 1], f=rk4_step): 
        """
        General integration function which calls a step function multiple times depending 
        on the parabolic integration strategy. 

        Parameters
        ----------
        u0 : array-like
            Initial condition of values
        dudt : function 
            Evolution function du/dt(t, u, ...)
        dt : float
            Time step
        T : (2, ) 
            Time range
        f : function
            Integration stepper function (e.g. RK4, EF, etc.)

        Returns
        -------
        t : (Nt, ) vector
            Time vector 
        u(t) : (Nt, ...) array-like
            Solution to the parabolic ODE. 
        """
        if u0 is None:
            u0=self.physics.f0
      
        dudt=self.physics.dfdx
        dt = self.physics.dx

        # T = [self.physics.x[0], self.physics.x[-2]]

        t = []
        ut = []

        u_n = u0  # initial condition
        t_n = T[0]
        
        while True: 
            ut.append(u_n)
            t.append(t_n)

            # update timestep
            t_n1 = t_n + dt
            if t_n1 > T[1] + dt/2:  # add some buffer here
                break
            u_n1 = f(t_n, u_n, dudt, dt)

            # update: 
            u_n = u_n1
            t_n = t_n1

        return np.array(t), np.array(ut)