"""
Integration object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np

class Integration():

    def __init__(self, physics, turbulence_model = None):
        """
        Args:
        physics (Physics object) - takes in a physics object to be integrated
        """

        self.physics = physics
        self.turbulence_model = turbulence_model


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

    def rk4_step1(t_n, u_n, dudt, dt, u=None, nuT=None): 
        """
        Computes the next timestep of u_n given the finite difference function du/dt
        with a 4-stage, 4th order accurate Runge-Kutta method. 
        
        Parameters
        ----------
        t_n : float
            time for time step n 
        u_n : array like
            condition at time step n 
        dudt : function 
            function du/dt(t, u) for 
        dt : float
            time step

        Returns u_(n+1)
        """    

        # step 1
        return dt * dudt(t_n, u_n, u=u, nuT=nuT)
    
    def rk4_step2(t_n, u_n, k1, dudt, dt, u=None, nuT=None): 
        """
        Computes the next timestep of u_n given the finite difference function du/dt
        with a 4-stage, 4th order accurate Runge-Kutta method. 
        
        Parameters
        ----------
        t_n : float
            time for time step n 
            must be the same for all integrated quantities
        u_n : list of arrays
            condition at time step n for integrated quantities
        dudt : list of function 
            function du/dt(t, u) for integrated quantities
        dt : float
            time step
            must be the same for all integrated quantities
        
        Returns u_(n+1)
        """    
        # step 2
        return dt * dudt(t_n + dt/2, u_n + k1/2, u=u, nuT=nuT)
    
    def rk4_step3(t_n, u_n, k2, dudt, dt, u=None, nuT=None): 
        """
        Computes the next timestep of u_n given the finite difference function du/dt
        with a 4-stage, 4th order accurate Runge-Kutta method. 
        
        Parameters
        ----------
        t_n : float
            time for time step n 
            must be the same for all integrated quantities
        u_n : list of arrays
            condition at time step n for integrated quantities
        dudt : list of function 
            function du/dt(t, u) for integrated quantities
        dt : float
            time step
            must be the same for all integrated quantities
        
        Returns u_(n+1)
        """    

        # step 3
        return dt * dudt(t_n + dt/2, u_n + k2/2, u=u, nuT=nuT)
        


    def rk4_step4(t_n, u_n, k1, k2, k3, dudt, dt, u=None, nuT=None): 
        """
        Computes the next timestep of u_n given the finite difference function du/dt
        with a 4-stage, 4th order accurate Runge-Kutta method. 
        
        Parameters
        ----------
        t_n : float
            time for time step n 
            must be the same for all integrated quantities
        u_n : list of arrays
            condition at time step n for integrated quantities
        dudt : list of function 
            function du/dt(t, u) for integrated quantities
        dt : float
            time step
            must be the same for all integrated quantities
        
        Returns u_(n+1)
        """    

 
        k4 = dt * dudt(t_n + dt, u_n + k3, u=u, nuT=nuT)

        return u_n + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        


    def EF_step(t_n, u_n, dudt, dt): 
        """
        Forward Euler stepping scheme
        """
        u_n1 = u_n + dt * dudt(t_n, u_n)
        return u_n1


    def integrate(self, u0=None, T=[0, 1], dudt=None, f=rk4_step): 
        """
        General integration function which calls a step function multiple times depending 
        on the parabolic integration strategy. 

        Parameters
        ----------
        u0 : array-like or list of arrays
            Initial condition of values
        dudt : function or list of functions
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
      
        if dudt is None:
            dudt=self.physics.dfdx

        dt = self.physics.dx

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

    def integrate_w_model(self, u0=None, k0=None, T=[0, 1], dudt=None): 
        """
        General integration function which calls a step function multiple times depending 
        on the parabolic integration strategy. 

        Parameters
        ----------
        u0 : array-like or list of arrays
            Initial condition of values
        dudt : function or list of functions
            Evolution function du/dt(t, u, ...)
        dt : float
            Time step
        T : (2, ) 
            Time range


        Returns
        -------
        t : (Nt, ) vector
            Time vector 
        u(t) : (Nt, ...) array-like
            Solution to the parabolic ODE. 
        """

        
        if u0 is None:
            u0=self.physics.f0
        
        if k0 is None:
            k0 = self.turbulence_model.f0
      
        if dudt is None:
            dudx = self.physics.dfdx
            dkdx = self.turbulence_model.dfdx

        dt = self.physics.dx

        t = []
        ut = []
        kt = []
        nuTt = []

        u_n = u0  # initial condition
        k_n = k0
        t_n = T[0]
        nuTt.append(self.turbulence_model.nuT[0,...])
        
        print(np.shape(k_n))
        i=0
        while True: 
            ut.append(u_n)
            kt.append(k_n)
            t.append(t_n)

            # update timestep
            t_n1 = t_n + dt
            if t_n1 > T[1] + dt/2:  # add some buffer here
                break
            # nuT = self.turbulence_model.nuT[i,...]
            nuT = self.turbulence_model.calculate_nuT(x=t_n, k=k_n, return_nuT=True)
            # self.physics.nuT = self.turbulence_model.nuT
            uk1 = dt * dudx(t_n, u_n, nuT=nuT)
            kk1 = dt * dkdx(t_n, k_n, u=u_n, nuT=nuT)

            nuT = self.turbulence_model.calculate_nuT(x=t_n + dt/2, k=k_n + kk1/2, return_nuT=True)
            
            uk2 = dt * dudx(t_n + dt/2, u_n + uk1/2, nuT=nuT)
            kk2 = dt * dkdx(t_n + dt/2, k_n + kk1/2, u=u_n + uk1/2, nuT=nuT)

            nuT = self.turbulence_model.calculate_nuT(x=t_n + dt/2, k=k_n + kk2/2, return_nuT=True)
            
            uk3 = dt * dudx(t_n + dt/2, u_n + uk2/2, nuT=nuT)
            kk3 = dt * dkdx(t_n + dt/2, k_n + kk2/2, u=u_n + uk2/2, nuT=nuT)

            nuT = self.turbulence_model.calculate_nuT(x=t_n + dt, k=k_n + kk3, return_nuT=True)
            
            uk4 = dt * dudx(t_n + dt, u_n + uk3, nuT=nuT)
            kk4 = dt * dkdx(t_n + dt, k_n + kk3, u=u_n + uk3, nuT=nuT)

            u_n1 = u_n + 1/6 * (uk1 + 2*uk2 + 2*uk3 + uk4)
            k_n1 = k_n + 1/6 * (kk1 + 2*kk2 + 2*kk3 + kk4)

            # update: 
            u_n = u_n1
            k_n = k_n1
            t_n = t_n1
            nuTt.append(nuT)

            i+=1

        return np.array(t), np.array(ut), np.array(kt), np.array(nuTt)
    