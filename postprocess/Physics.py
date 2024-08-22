"""
Physics object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np
from typing import Optional
from utils import finite_diff, derivative_z_ghost_cells
from padeopsIO import budget_utils
from postprocess import TurbulenceModel

class Physics():

    def __init__(self, 
                io: Optional[object] = None,
                padeops: bool=False, 
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

        self.init_grid()
        self.create_terms_dict()  
        self.get_ic(self.x[0])

    def init_grid(self):
        """
        Initialize the grid
        """

        if self.padeops:
            xid, yid, zid = self.io.get_xids(x=self.xlim, y=self.ylim, z=self.zlim, return_none=True, return_slice=True)

            self.x = self.io.xLine[xid]
            self.y = self.io.yLine[yid]
            self.z = self.io.zLine[zid]
            
            self.xid = xid
            self.yid = yid
            self.zid = zid

        else:
            self.x = np.arange(np.min(self.xlim), np.max(self.xlim) + self.dx, self.dx)
            self.y = np.arange(np.min(self.ylim), np.max(self.ylim) + self.dy, self.dy)
            self.z = np.arange(np.min(self.zlim), np.max(self.zlim) + self.dz, self.dz)

    def create_terms_dict(self):
        """
        Creates a dictionary of terms to be passed to dfdx
        """
        pass
            

    def get_ic(self, x0):
        """
        Returns the initial condition from the x0 location
        """
        xid = np.argmin(np.abs(self.x - x0))
        self.f0 = self.f[xid,...]

    def dfdx(self, x, f): 
            """du/dt function"""

            xid = np.argmin(np.abs(self.x - x))

            dfdy = finite_diff(f.T, self.dy).T
            dfdz = finite_diff(f, self.dz)

            rhs = 0
            for key, val in self.rhs_terms.items():
                # print(key)
                rhs += val[xid,...]

            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/self.u[xid,...]
        
class WakeDeficitX(Physics):
    """
    Class that extends Physics to model streamwise wake deficit
    """
    
    def __init__(self, io,
                 padeops: bool = True, 
                 LES_dpdx: bool = True,
                 LES_delta_v_delta_w_adv: bool = True,
                 LES_xturb: bool = True,
                 LES_xCor: bool = True,
                 LES_xSGS: bool = True,
                 LES_xturb_x: bool = False,
                 LES_xturb_y: bool = False,
                 LES_xturb_z: bool = False,
                 turbulence_model: str = None, 
                 base_io:Optional[object] = None,
                 prim_io:Optional[object] = None,               
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

        self.base_io = base_io
        self.prim_io = prim_io
        self.LES_dpdx = LES_dpdx
        self.LES_delta_v_delta_w_adv = LES_delta_v_delta_w_adv
        self.LES_xturb = LES_xturb
        self.LES_xCor = LES_xCor
        self.LES_xSGS = LES_xSGS
        self.LES_xturb_x = LES_xturb_x
        self.LES_xturb_y = LES_xturb_y
        self.LES_xturb_z = LES_xturb_z
  

        self.turbulence_model = turbulence_model

        super().__init__(io, padeops, xlim, ylim, zlim, dx, dy, dz, Uref, Lref)         

    def create_terms_dict(self):
        """
        Create the necessary dictionary of rhs terms
        """
        self.rhs_terms = {}

        if self.padeops:
            budget_terms = ['delta_u', 'delta_v', 'delta_w', 'dpdx', 'xAdv_base_delta_fluc', 
                'xAdv_delta_delta_fluc', 'xAdv_delta_base_fluc', 'xCor', 'xAdv_delta_base_mean', 'xSGS']
            
            if not all(item in self.io.budget for item in budget_terms):
                self.io.read_budgets(budget_terms)
            if not all(item in self.base_io.budget for item in ['ubar', 'vbar', 'wbar']):
                self.base_io.read_budgets(['ubar', 'vbar', 'wbar'])

            xid = self.xid  
            yid = self.yid
            zid = self.zid

            nonDim = self.Lref/(self.Uref**2)

            if self.LES_dpdx:
                self.rhs_terms['dpdx'] = self.io.budget['dpdx'][xid,yid,zid] * nonDim

            if self.LES_delta_v_delta_w_adv:
                self.rhs_terms['delta_v_delta_w_adv'] = (self.io.budget['xAdv_delta_base_mean'][xid,yid,zid]) * nonDim

            if self.LES_xturb:
                self.rhs_terms['xturb'] = (self.io.budget['xAdv_delta_delta_fluc'][xid,yid,zid] + self.io.budget['xAdv_delta_base_fluc'][xid,yid,zid] + self.io.budget['xAdv_base_delta_fluc'][xid,yid,zid]) * nonDim
            else:
                if self.LES_xturb_x:
                    if 'uu_wake' not in self.io.budget:
                        self.base_io.read_budgets(['uu'])
                        self.prim_io.read_budgets(['uu'])

                        self.io.budget['uu_wake'] = self.prim_io.budget['uu'] - self.base_io.budget['uu']
                    
                    self.rhs_terms['xturb_x'] = - np.gradient(self.io.budget['uu_wake'], self.dx, axis=0)[xid,yid,zid]/self.Uref**2
                
                if self.LES_xturb_y:
                    if 'uv_wake' not in self.io.budget:
                        self.base_io.read_budgets(['uv'])
                        self.prim_io.read_budgets(['uv'])

                        self.io.budget['uv_wake'] = self.prim_io.budget['uv'] - self.base_io.budget['uv']
                    
                    self.rhs_terms['xturb_y'] = - np.gradient(self.io.budget['uv_wake'], self.dy, axis=1)[xid,yid,zid]/self.Uref**2

                if self.LES_xturb_z:
                    if 'uw_wake' not in self.io.budget:
                        self.base_io.read_budgets(['uw'])
                        self.prim_io.read_budgets(['uw'])

                        self.io.budget['uw_wake'] = self.prim_io.budget['uw'] - self.base_io.budget['uw']
                    
                    self.rhs_terms['xturb_z'] = - np.gradient(self.io.budget['uw_wake'], self.dz, axis=2)[xid,yid,zid]/self.Uref**2
                
                if self.turbulence_model is not None:
                    if self.turbulence_model == "exact_rs":
                        if not all(item in self.io.budget for item in ['uu_wake', 'uv_wake', 'uw_wake']):
                            if not all(item in self.base_io.budget for item in ['uu', 'uv', 'uw']):
                                self.base_io.read_budgets(['uu', 'uv', 'uw'])
                            if not all(item in self.prim_io.budget for item in ['uu', 'uv', 'uw']):
                                self.prim_io.read_budgets(['uu', 'uv', 'uw'])
                                
                            self.io.budget['uu_wake'] = self.prim_io.budget['uu'] - self.base_io.budget['uu']
                            self.io.budget['uv_wake'] = self.prim_io.budget['uv'] - self.base_io.budget['uv']
                            self.io.budget['uw_wake'] = self.prim_io.budget['uw'] - self.base_io.budget['uw']

                        self.rhs_terms['xturb'] = (- np.gradient(self.io.budget['uu_wake'], self.dx, axis=0) \
                                                    - np.gradient(self.io.budget['uv_wake'], self.dy, axis=1) \
                                                    - np.gradient(self.io.budget['uw_wake'], self.dz, axis=2))[xid,yid,zid] / self.Uref**2
                    else:
                        self.rhs_terms['xturb_model'] = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))
                        self.nuT = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))
                        self.nuT_base = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))
                        self.nuT_y = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))

            if self.LES_xCor:
                self.rhs_terms['xCor'] = self.io.budget['xCor'][xid,yid,zid] * nonDim
            
            if self.LES_xSGS:
                self.rhs_terms['xSGS'] = self.io.budget['xSGS'][xid,yid,zid] * nonDim

            self.f = self.io.budget['delta_u'][xid,yid,zid]/self.Uref

            self.u = (self.base_io.budget['ubar'])[xid,yid,zid]/self.Uref
            if self.LES_delta_v_delta_w_adv:
                self.v = (self.base_io.budget['vbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'] + self.io.budget['delta_w'])[xid,yid,zid]/self.Uref
            else:
                self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref

    def dfdx(self, x, f): 
            """du/dt function for delta u"""
            xid = np.argmin(np.abs(self.x - x))
            u = self.u[xid,...] + f

            # dfdy = np.gradient(f, self.dy, axis=0)
            # dfdz = np.gradient(f, self.dz, axis=1)

            # dUdy = np.gradient(self.u[xid,...], self.dy, axis=0)
            # dUdz = np.gradient(self.u[xid,...], self.dz, axis=1)

            dfdy = finite_diff(f.T, self.dy).T
            dfdz = finite_diff(f, self.dz)

            rhs = 0
            for key, val in self.rhs_terms.items():
                if 'model' in key:
                    if self.turbulence_model == 'scott_nonlinear':
                        dUbdy = finite_diff(self.u[xid,...].T, self.dy).T  
                        dUbdz = finite_diff(self.u[xid,...], self.dz)
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT[xid,...] * (dfdy + dUbdy)).T, self.dy).T \
                            + finite_diff(self.nuT[xid,...] * (dfdz + dUbdz), self.dz) \
                            + finite_diff((self.nuT_base[xid,...] * dfdy).T, self.dy).T \
                            + finite_diff(self.nuT_base[xid,...] * dfdz, self.dz)
                        
                    elif self.turbulence_model == 'scott_nuT_y': 
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT_y[xid,...] * (dfdy)).T, self.dy).T \
                            + finite_diff(self.nuT[xid,...] * (dfdz), self.dz)
                        
                    elif self.turbulence_model == 'scott_nuT_y_only':
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT_y[xid,...] * (dfdy)).T, self.dy).T

                    elif self.turbulence_model == 'scott_nuT_z_only':
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff(self.nuT[xid,...] * (dfdz), self.dz)

                    else:
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT[xid,...] * (dfdy)).T, self.dy).T \
                                + finite_diff(self.nuT[xid,...] * (dfdz), self.dz)

                rhs += val[xid,...]
                # rhs += val[xid,...]

            # dfdz = self.dfdz[xid,...]
            # dfdy = self.dfdy[xid,...]
            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/u


class WakeTKE(Physics):
    """
    Class that extends Physics to model streamwise wake deficit
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
        self.base_io = base_io
        self.prim_io = prim_io

        super().__init__(io, padeops, xlim, ylim, zlim, dx, dy, dz, Uref, Lref)         

    def create_terms_dict(self):
        """
        Create the necessary dictionary of rhs terms
        """
        self.rhs_terms = {}

        if self.padeops:
            # budget_terms = ['delta_u', 'delta_v', 'delta_w', 'TKE_shear_production_wake', 'TKE_buoyancy_wake', 
            #     'TKE_dissipation_wake', 'TKE_turb_transport_wake', 'TKE_turb_transport_wake', 'TKE_SGS_transport_wake', 'xSGS']
            
            # self.io.read_budgets(budget_terms)
            self.base_io.read_budgets(['ubar', 'vbar', 'wbar', 'uu', 'vv', 'ww'])
            self.prim_io.read_budgets(['uu', 'vv', 'ww'])
            self.base_io.read_budgets('budget3')
            self.prim_io.read_budgets('budget3')

            self.base_io.budget['TKE'] = 0.5 * (self.base_io.budget['uu'] + self.base_io.budget['vv'] + self.base_io.budget['ww'])
            self.prim_io.budget['TKE'] = 0.5 * (self.prim_io.budget['uu'] + self.prim_io.budget['vv'] + self.prim_io.budget['ww'])

            self.io.budget['TKE_wake'] = self.prim_io.budget['TKE'] - self.base_io.budget['TKE']

            budget_terms = [term for term in self.prim_io.key if self.prim_io.key[term][0] == 3]
            for term in budget_terms:
                self.io.budget[term + '_wake'] = self.prim_io.budget[term] - self.base_io.budget[term]

            xid = self.xid  
            yid = self.yid
            zid = self.zid

            nonDim = self.Lref/(self.Uref**3)

            if self.LES_prod:
                self.rhs_terms['prod'] = self.io.budget['TKE_shear_production_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['prod'] = np.zeros(np.shape(self.io.budget['TKE_shear_production_wake'][xid,yid,zid]))

            if self.LES_buoy:
                self.rhs_terms['buoy'] = self.io.budget['TKE_buoyancy_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['buoy'] = np.zeros(np.shape(self.io.budget['TKE_buoyancy_wake'][xid,yid,zid]))

            if self.LES_diss:
                self.rhs_terms['diss'] = self.io.budget['TKE_dissipation_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['diss'] = np.zeros(np.shape(self.io.budget['TKE_dissipation_wake'][xid,yid,zid]))

            if self.LES_turb_trans:
                self.rhs_terms['turb_trans'] = self.io.budget['TKE_turb_transport_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['turb_trans'] = np.zeros(np.shape(self.io.budget['TKE_turb_transport_wake'][xid,yid,zid]))
            
            if self.LES_pres_trans:
                self.rhs_terms['pres_trans'] = self.io.budget['TKE_p_transport_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['pres_trans'] = np.zeros(np.shape(self.io.budget['TKE_p_transport_wake'][xid,yid,zid]))

            if self.LES_sgs_trans:
                self.rhs_terms['sgs_trans'] = self.io.budget['TKE_SGS_transport_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['sgs_trans'] = np.zeros(np.shape(self.io.budget['TKE_SGS_transport_wake'][xid,yid,zid]))

            if self.LES_sgs_trans:
                self.rhs_terms['sgs_trans'] = self.io.budget['TKE_SGS_transport_wake'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['sgs_trans'] = np.zeros(np.shape(self.io.budget['TKE_SGS_transport_wake'][xid,yid,zid]))

            if self.LES_delta_v_delta_w_adv:
                self.rhs_terms['delta_v_delta_w_adv_b'] = budget_utils.advection([self.io.budget['delta_u'],self.io.budget['delta_v'],self.io.budget['delta_w']],
                                                                                     self.base_io.budget['TKE'], self.dx, self.dy, self.dz)[xid,yid,zid] * nonDim
            else:
                self.rhs_terms['delta_v_delta_w_adv_b'] = np.zeros(np.shape(self.io.budget['TKE_SGS_transport_wake'][xid,yid,zid]))

            self.f = self.io.budget['TKE_wake'][xid,yid,zid]/self.Uref**2

            self.u = (self.base_io.budget['ubar'] + self.io.budget['delta_u'])[xid,yid,zid]/self.Uref
            if self.LES_delta_v_delta_w_adv:
                self.v = (self.base_io.budget['vbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'] + self.io.budget['delta_w'])[xid,yid,zid]/self.Uref
            else:
                self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref

            # #TEST
            # self.dfdz = np.gradient(self.f, self.dz, axis=2)
            # self.dfdy = np.gradient(self.f, self.dy, axis=1)






