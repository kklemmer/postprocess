"""
Turbulence model object for forward marching 

Extends the Physics class

Kerry Klemmer
August 2024
"""
import numpy as np
from typing import Optional
from scipy import ndimage
from padeopsIO import budget_utils
from utils import finite_diff

from postprocess.Integration import Integration

class WakeModel():

    def __init__(self, delta_u, 
                 turbulence_model: Optional[object] = None):

        self.delta_u = delta_u
        self.turbulence_model = turbulence_model

        # initialize integration objects for both
        self.delta_u_int = Integration(delta_u)
        # self.turbulence_model_int = Integration(turbulence_model)
             
        
    def calculate_wake(self):
        """
        forward marches the wake in x
        """

        if self.turbulence_model is None:
            x, du = self.delta_u_int.integrate(T=[self.delta_u.x[0], self.delta_u.x[-1]])

            self.du = du
            return
        elif "scott" in self.delta_u.turbulence_model:
            self.delta_u.nuT = self.turbulence_model.nuT
            if self.delta_u.turbulence_model == 'scott_nonlinear':
                self.delta_u.nuT_base = self.turbulence_model.nuT_base
            
            if self.delta_u.turbulence_model == 'scott_nuT_y':
                self.delta_u.nuT_y = self.turbulence_model.nuT_y
            
            x, du = self.delta_u_int.integrate(T=[self.delta_u.x[0], self.delta_u.x[-1]])

            self.du = du
        else:
            # step 1 get nuT IC
            self.turbulence_model.get_ic(self.turbulence_model.x[0])
            # share this with delta_u
            self.delta_u.nuT = self.turbulence_model.nuT
            if self.delta_u.turbulence_model == 'scott_nonlinear':
                self.delta_u.nuT_base = self.turbulence_model.nuT_base
            
            if self.delta_u.turbulence_model == 'scott_nuT_y':
                self.delta_u.nuT_y = self.turbulence_model.nuT_y
            
            du = []
            tmp_du_old = np.copy(self.delta_u.f0)
            du.append(tmp_du_old)


        

            for i in range(len(self.delta_u.x)-1):

                # step 2 calculate delta u at x + dx
                tmp_x, tmp = self.delta_u_int.integrate(u0=tmp_du_old, T=[self.delta_u.x[i], self.delta_u.x[i+1]])
                tmp_du = np.squeeze(tmp)[1]
                du.append(tmp_du)

                # step 3 calculate nuT at x + dx
                # self.turbulence_model.calculate_nuT(x=self.delta_u.x[i+1], u=tmp_du)
                # self.calculate_rs_div(x=self.delta_u.x[i+1], u=tmp_du)


                # xid = np.argmin(np.abs(self.delta_u.x - self.delta_u.x[i+1]))

                # step 4 share new nuT with delta_u
                self.delta_u.nuT = self.turbulence_model.nuT

                if self.delta_u.turbulence_model == 'scott_nonlinear':
                    self.delta_u.nuT_base = self.turbulence_model.nuT_base

                if self.delta_u.turbulence_model == 'scott_nuT_':
                    self.delta_u.nuT_y = self.turbulence_model.nuT_y

                tmp_du_old = tmp_du


            self.du = np.array(du)

    def calculate_rs_div(self, x, u):
        """
        Calculate the Reynolds stress divergence
        """
        xid = np.argmin(np.abs(self.delta_u.x - x))

        turbulence_model = self.delta_u.turbulence_model
        nuT = self.turbulence_model.nuT
        dy = self.delta_u.dy
        dz = self.delta_u.dz

        if turbulence_model == "scott_nonlinear":
            dudy = finite_diff(u.T, dy).T
            dudz = finite_diff(u, dz)
            dUbdy = finite_diff(self.delta_u.u[xid,...].T, dy).T
            dUbdz = finite_diff(self.delta_u.u[xid,...], dz)
            nuT_base = self.turbulence_model.nuT_base

            self.delta_u.rhs_terms['xturb_model'][xid,...] = finite_diff((nuT[xid,...] * (dudy + dUbdy)).T, dy).T \
                            + finite_diff(nuT[xid,...] * (dudz + dUbdz), dz) \
                            + finite_diff((nuT_base[xid,...] * dudy).T, dy).T \
                            + finite_diff(nuT_base[xid,...] * dudz, dz)
        elif turbulence_model == "scott_nuT_y":
            dudy = finite_diff(u.T, dy).T
            dudz = finite_diff(u, dz)

            self.delta_u.rhs_terms['xturb_model'][xid,...] = finite_diff((self.turbulence_model.nuT_y[xid,...] * dudy).T, dy).T \
                            + finite_diff(nuT[xid,...] * dudz, dz)
        else:
            dudy = finite_diff(u.T, dy).T
            dudz = finite_diff(u, dz)

            self.delta_u.rhs_terms['xturb_model'][xid,...] = finite_diff((nuT[xid,...] * dudy).T, dy).T \
                            + finite_diff(nuT[xid,...] * dudz, dz)            

    def calculate_ud(self, xloc, LES=False):
        """
        Calculate the disk averaged velocity ud within xlim
        """

        xG, yG, zG = np.meshgrid(self.delta_u.x, self.delta_u.y, self.delta_u.z, indexing='ij')  # creates 3D meshgrid
        xT, yT, zT = (xloc, 0, 0)  # coordinates of downwind turbine
        R = 0.5
        thick = self.delta_u.dx
        kernel = ((yG - yT)**2 + (zG - zT)**2 < R**2) * (abs(xG - xT) < thick)
        kernel_normalized = kernel / np.sum(kernel)

        if not LES:
            ud = np.sum(self.du * kernel_normalized)
        else:
            ud = np.sum(self.delta_u.f * kernel_normalized)

        return ud


        