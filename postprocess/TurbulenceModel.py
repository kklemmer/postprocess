"""
Turbulence model object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np
from typing import Optional
from scipy import ndimage
from padeopsIO import budget_utils
from postprocess.Physics import Physics
from postprocess.Physics import WakeTKE

from scipy.stats import linregress



class TurbulenceModel():

    init_grid = Physics.__dict__["init_grid"]

    def __init__(self, io,
                padeops: bool=False, 
                base_io: Optional[object] = None,
                prim_io: Optional[object] = None, 
                xlim: Optional[list] = None,
                ylim: Optional[list] = None,
                zlim: Optional[list] = None,
                dx: float = 0.1,
                dy: float = 0.1,
                dz: float = 0.1,
                Uref: float = 1.,
                Lref: float = 1.):

        self.io = io
        self.padeops = padeops
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.Uref = Uref
        self.Lref = Lref
        
        if self.padeops:
            self.dx = io.dx
            self.dy = io.dy
            self.dz = io.dz
        else:
            self.dx = dx
            self.dy = dy
            self.dz = dz

        if padeops:
            self.base_io = base_io
            self.prim_io = prim_io

        self.init_grid()
        

    def get_ic(self, x0):
        """
        Get initial condition
        """
        pass
        return
        
    def calculate_nuT(self, x, **kwargs):
        """
        Calculate nuT at a given x location
        """

        pass

    def calculate_nuT_exact(self):
        """
        Calculate the eddy viscosity from LES based on <u'w'>_wake and Delta S13
        """

        xid = self.xid
        yid = self.yid
        zid = self.zid

        self.prim_io.read_budgets(['uw'])
        self.base_io.read_budgets(['uw'])
        self.prim_io.read_budgets(['uv'])
        self.base_io.read_budgets(['uv'])

        self.io.budget['uw_wake'] = self.prim_io.budget['uw'] - self.base_io.budget['uw']
        self.io.budget['uv_wake'] = self.prim_io.budget['uv'] - self.base_io.budget['uv']

        self.io.budget['dudz'] = np.gradient(self.io.budget['delta_u'], self.dz, axis=2)
        self.io.budget['dudy'] = np.gradient(self.io.budget['delta_u'], self.dy, axis=1)

        self.nuT_exact = (-self.io.budget['uw_wake'] / self.io.budget['dudz'])[xid,yid,zid] / self.Uref


class nuT_exact_3d(TurbulenceModel):
    """
    k-l turbulence model
    """

    def __init__(self, io,
                 padeops: bool = True, 
                 base_io: Optional[object] = None,
                 prim_io: Optional[object] = None,               
                 xlim: Optional[list] = None,
                 ylim: Optional[list] = None,
                 zlim: Optional[list] = None,
                 dx: float = 0.1,
                 dy: float = 0.1,
                 dz: float = 0.1,
                 Uref: float = 1.,
                 Lref: float = 1.):
        """
        Calls the constructor of Physics
        """

        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref) 

        self.calculate_nuT_exact()
        self.nuT = self.nuT_exact


    def calculate_nuT(self, x, **kwargs):
        """
        Calculate nuT at a given x location
        """

        pass

        return
    

class kl_TurbulenceModel(WakeTKE, TurbulenceModel):
    """
    k-l turbulence model
    """

    def __init__(self, io,
                 padeops: bool = True, 
                 LES_prod: bool = True,
                 LES_buoy: bool = True,
                 LES_diss: bool = True,
                 LES_turb_trans: bool = True,
                 LES_pres_trans: bool = True, 
                 LES_sgs_trans: bool = True,
                 LES_delta_v_delta_w_adv: bool = True,
                 base_io: Optional[object] = None,
                 prim_io: Optional[object] = None,               
                 xlim: Optional[list] = None,
                 ylim: Optional[list] = None,
                 zlim: Optional[list] = None,
                 dx: float = 0.1,
                 dy: float = 0.1,
                 dz: float = 0.1,
                 Uref: float = 1.,
                 Lref: float = 1.):
        """
        Calls the constructor of Physics
        """
        self.LES_prod = LES_prod
        self.LES_buoy = LES_buoy
        self.LES_diss = LES_diss
        self.LES_turb_trans = LES_turb_trans
        self.LES_pres_trans = LES_pres_trans 
        self.LES_sgs_trans = LES_sgs_trans
        self.LES_delta_v_delta_w_adv = LES_delta_v_delta_w_adv

        super().__init__(io=io, padeops=padeops, base_io=base_io, prim_io=prim_io, xlim=xlim, ylim=ylim, zlim=zlim, dx=dx, dy=dy, dz=dz, Uref=Uref, Lref=Lref) 

        self.calculate_nuT_exact()
        self.ell = self.nuT_exact / np.sqrt(self.f)
        self.ell[np.isnan(self.ell)] = 0

        self.tke = np.zeros(np.shape(self.f))
        self.tke[0,...] = self.f0

        self.nuT = np.zeros(np.shape(self.f))
        self.nuT[0,...] = self.nuT_exact[0,...]

    def calculate_nuT(self, x, u):
        """
        Calculate the eddy viscosity nuT at a given x location for the k-l model
        """
        xid = np.argmin(np.abs(self.x - x))
        if xid == 0:
            return
        self.u[xid,...] = u

        def _integrate_tke(x):

            xid = np.argmin(np.abs(self.x - x))

            def _rk4_step(t_n, u_n, dudt, dt): 
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

            tke_n1 = _rk4_step(x, self.tke[xid,...], self.dfdx, self.dx)

            return tke_n1
        
        
        tke_n1 = _integrate_tke(x)
        self.tke[xid,...] = tke_n1
        self.nuT[xid,...] = np.sqrt(tke_n1) * self.ell_y[xid,...]
        self.nuT[np.isnan(self.nuT)] = 0


class scott_data_TurbulenceModel(TurbulenceModel):
    """
    Eddy viscosity based on the data-driven turbulence 
    modeling procedure outlined in Scott et al. 2023
    """

    def __init__(self, io,
                 padeops: bool = True, 
                 base_io: Optional[object] = None,
                 prim_io: Optional[object] = None,               
                 xlim: Optional[list] = None,
                 ylim: Optional[list] = None,
                 zlim: Optional[list] = None,
                 dx: float = 0.1,
                 dy: float = 0.1,
                 dz: float = 0.1,
                 Uref: float = 1.,
                 Lref: float = 1.,
                 calculate_nuT_y: bool = False,
                 streamtube: bool = False, 
                 Cz: float = 1):

        self.calculate_nuT_y = calculate_nuT_y
        self.streamtube = streamtube
        self.Cz = Cz

        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref) 

        self.calculate_nuT_slope()

    def calculate_nuT_slope(self):
        """
        Calculate the 1D nuT following the procedure in Scott et al 2023
        """

        slopes = np.zeros(len(self.x))
        slope_err = np.zeros(len(self.x))

        if 'uw_wake' not in self.io.budget:
            if 'uw' not in self.base_io.budget:
                self.base_io.read_budgets(['uw'])
            if 'uw' not in self.prim_io.budget:
                self.prim_io.read_budgets(['uw'])

            self.io.budget['uw_wake'] = self.prim_io.budget['uw'] - self.base_io.budget['uw']

        # if 'S13' not in self.io.budget:
        self.io.budget['S13'] = 0.5 * (np.gradient(self.io.budget['delta_u'], self.dz, axis=2) \
                                           + np.gradient(self.io.budget['delta_w'], self.dx, axis=0))

        # if 'S13' not in self.prim_io.budget:
        #     self.prim_io.budget['S13'] = 0.5 * (np.gradient(self.prim_io.budget['ubar'], self.dz, axis=2) \
        #                                    + np.gradient(self.prim_io.budget['vbar'], self.dx, axis=0))
            
        if self.streamtube:
            x_data = self.io.budget['S13'] * self.prim_io.stream_mask
            y_data = self.io.budget['uw_wake'] * self.prim_io.stream_mask
        else:
            x_data = self.io.budget['S13'][self.xid, self.yid, self.zid]
            y_data = self.io.budget['uw_wake'][self.xid, self.yid, self.zid]
            
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT = self.Cz * slopes / self.Uref
        self.nuT_err = self.Cz * slope_err / self.Uref

        return


class scott_data_nonlinear_TurbulenceModel():
    """
    Eddy viscosity based on the data-driven turbulence 
    modeling procedure outlined in Scott et al. 2023

    Here we treat the Boussinesq model linearly 
    TODO: add details
    """

    def __init__(self, io,
                 padeops: bool = True, 
                 base_io: Optional[object] = None,
                 prim_io: Optional[object] = None,               
                 xlim: Optional[list] = None,
                 ylim: Optional[list] = None,
                 zlim: Optional[list] = None,
                 dx: float = 0.1,
                 dy: float = 0.1,
                 dz: float = 0.1,
                 Uref: float = 1.,
                 Lref: float = 1.):

        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref) 

        self.calculate_nuT_slope()
        self.calculate_nuT_base_slope()


    def calculate_nuT_base_slope(self):
        """
        Calculate the 1D nuT following the procedure in Scott et al 2023
        """

        slopes = np.zeros(len(self.x))
        slope_err = np.zeros(len(self.x))

        if 'uw' not in self.base_io.budget:
                self.base_io.read_budgets(['uw'])

        if 'S13' not in self.base_io.budget:
            self.base_io.budget['S13'] = 0.5 * (np.gradient(self.base_io.budget['ubar'], self.dz, axis=2) \
                                           + np.gradient(self.base_io.budget['vbar'], self.dx, axis=0))
            
        # wake only component first    
        x_data = self.base_io.budget['S13'][self.xid, self.yid, self.zid]
        y_data = self.base_io.budget['uw'][self.xid, self.yid, self.zid]
        
        
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT_base = slopes / self.Uref
        self.nuT_base_err = slope_err / self.Uref

        return

    def calculate_nuT_slope(self):
        """
        Calculate the 1D nuT for delta delta only following the procedure in Scott et al 2023
        """

        slopes = np.zeros(len(self.x))
        slope_err = np.zeros(len(self.x))

        if 'delta_uw' not in self.io.budget:
            self.io.read_budgets(['delta_uw'])

        # if 'S13' not in self.io.budget:
        self.io.budget['S13'] = 0.5 * (np.gradient(self.io.budget['delta_u'], self.dz, axis=2) \
                                           + np.gradient(self.io.budget['delta_w'], self.dx, axis=0))

        # if 'S13' not in self.prim_io.budget:
        #     self.prim_io.budget['S13'] = 0.5 * (np.gradient(self.prim_io.budget['ubar'], self.dz, axis=2) \
        #                                    + np.gradient(self.prim_io.budget['vbar'], self.dx, axis=0))
            
        if self.streamtube:
            x_data = self.io.budget['S13'] * self.prim_io.stream_mask
            y_data = self.io.budget['delta_uw'] * self.prim_io.stream_mask
        else:
            x_data = self.io.budget['S13'][self.xid, self.yid, self.zid]
            y_data = self.io.budget['delta_uw'][self.xid, self.yid, self.zid]
            
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT = slopes / self.Uref
        self.nuT_err = slope_err / self.Uref

        return

class scott_data_nuT_y_TurbulenceModel(scott_data_TurbulenceModel):
    """
    Eddy viscosity based on the data-driven turbulence 
    modeling procedure outlined in Scott et al. 2023
    """

    def __init__(self, io,
                 padeops: bool = True, 
                 base_io: Optional[object] = None,
                 prim_io: Optional[object] = None,               
                 xlim: Optional[list] = None,
                 ylim: Optional[list] = None,
                 zlim: Optional[list] = None,
                 dx: float = 0.1,
                 dy: float = 0.1,
                 dz: float = 0.1,
                 Uref: float = 1.,
                 Lref: float = 1.,
                 Cy: float = 1,
                 Cz: float = 1):

        self.Cy = Cy

        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref, Cz=Cz) 

        self.calculate_nuT_slope()
        self.calculate_nuT_y_slope()

    def calculate_nuT_y_slope(self):
        """
        Calculate the 1D nuT following the procedure in Scott et al 2023
        """

        if 'uv_wake' not in self.io.budget:
            if 'uv' not in self.base_io.budget:
                self.base_io.read_budgets(['uv'])
            if 'uv' not in self.prim_io.budget:
                self.prim_io.read_budgets(['uv'])

            self.io.budget['uv_wake'] = self.prim_io.budget['uv'] - self.base_io.budget['uv']

        # if 'S13' not in self.io.budget:
        self.io.budget['S12'] = 0.5 * (np.gradient(self.io.budget['delta_u'], self.dy, axis=1) \
                                        + np.gradient(self.io.budget['delta_v'], self.dx, axis=0))

        # if 'S13' not in self.prim_io.budget:
        #     self.prim_io.budget['S13'] = 0.5 * (np.gradient(self.prim_io.budget['ubar'], self.dz, axis=2) \
        #                                    + np.gradient(self.prim_io.budget['vbar'], self.dx, axis=0))
            
        if self.streamtube:
            x_data = self.io.budget['S12'] * self.prim_io.stream_mask
            y_data = self.io.budget['uv_wake'] * self.prim_io.stream_mask
        else:
            x_data = self.io.budget['S12'][self.xid, self.yid, self.zid]
            y_data = self.io.budget['uv_wake'][self.xid, self.yid, self.zid]
            
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT_y = self.Cy * slopes / self.Uref
        self.nuT_y_err = self.Cy * slope_err / self.Uref

        return

def calculate_slope(x_data, y_data):

    slopes = np.zeros(np.shape(x_data)[0])
    slope_err = np.zeros(np.shape(x_data)[0])

    for i in range(len(slopes)):
        slope, intercept, r_value, p_value, std_err = linregress(-x_data[i].ravel(), y_data[i].ravel())
        slopes[i] = slope
        slope_err[i] = std_err

    return slopes, slope_err