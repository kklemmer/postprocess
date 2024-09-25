"""
Physics object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np
from typing import Optional
from postprocess.utils import finite_diff, wall_bc, top_bc
from padeopsIO import budget_utils
from scipy.signal import convolve2d


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

            self.nx = len(self.x)
            self.ny = len(self.y)
            self.nz = len(self.z)

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

    def dfdx(self, x, f, u=None, nuT=None): 
            """du/dt function"""

            xid = np.argmin(np.abs(self.x - x))

            if u is None:
                u = self.u[xid,...]

            dfdy = finite_diff(f.T, self.dy).T
            dfdz = finite_diff(f, self.dz)

            rhs = 0
            for key, val in self.rhs_terms.items():
                rhs += val[xid,...]

            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/u
        
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
                 Lref: float = 1., 
                 model_ic: bool = False,
                 veer_correction: bool = False,
                 dv_dw_streamtube: bool = False):
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
        self.model_ic = model_ic

        self.d = 1

        self.turbulence_model = turbulence_model

        self.veer_correction = veer_correction
        self.dv_dw_streamtube = dv_dw_streamtube

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

            self.u = self.base_io.budget['ubar'][xid,yid,zid]/self.Uref
            self.du = self.io.budget['delta_u'][xid,yid,zid]/self.Uref
            self.ub = self.base_io.budget['ubar'][xid,yid,zid]/self.Uref
            if self.LES_delta_v_delta_w_adv:
                self.v = (self.base_io.budget['vbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'] + self.io.budget['delta_w'])[xid,yid,zid]/self.Uref

                self.dv = self.io.budget['delta_v'][xid,yid,zid]/self.Uref
                self.vb = self.base_io.budget['vbar'][xid,yid,zid]/self.Uref
                self.dw = self.io.budget['delta_w'][xid,yid,zid]/self.Uref
                self.wb = self.base_io.budget['wbar'][xid,yid,zid]/self.Uref

                if self.dv_dw_streamtube:
                    dv = np.sum(self.io.budget['delta_v'][xid,yid,zid]/self.Uref * self.prim_io.stream_mask[xid,yid,zid], axis=(1,2)) \
                                / np.sum(self.prim_io.stream_mask[xid,yid,zid], axis=(1,2))
                    self.dv = np.broadcast_to(dv[:,np.newaxis,np.newaxis], (self.nx, self.ny, self.nz))

                    dw = np.sum(self.io.budget['delta_w'][xid,yid,zid]/self.Uref * self.prim_io.stream_mask[xid,yid,zid], axis=(1,2)) \
                                / np.sum(self.prim_io.stream_mask[xid,yid,zid], axis=(1,2)) 
                    self.dw = np.broadcast_to(dw[:,np.newaxis,np.newaxis], (self.nx, self.ny, self.nz))

                    self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref + self.dv
                    self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref + self.dw
            else:
                self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref

                self.dv = np.zeros(np.shape(self.io.budget['delta_v'][xid,yid,zid]))
                self.vb = self.base_io.budget['vbar'][xid,yid,zid]/self.Uref
                self.dw = np.zeros(np.shape(self.io.budget['delta_v'][xid,yid,zid]))
                self.wb = self.base_io.budget['wbar'][xid,yid,zid]/self.Uref

                if self.veer_correction:
                    zh = np.argmin(np.abs(self.z))
                    z_top = np.argmin(np.abs(self.z-0.5))
                    z_bot = np.argmin(np.abs(self.z+0.5))
                    self.N = z_top - z_bot + 1

                    wd_top = np.rad2deg(np.arctan(np.mean(self.vb[...,z_top]/self.ub[...,z_top])))
                    wd_bot = np.rad2deg(np.arctan(np.mean(self.vb[...,z_bot]/self.ub[...,z_bot])))

                    self.yaw = np.abs(wd_top - wd_bot)
                    self.yaw = 5

                    self._compute_vw()

                    self.v = self.vb + self.dv
                    self.w = self.wb + self.dw



    def get_ic(self, x0, smooth_fact=1.5):
        """
        Returns the initial condition from the x0 location
        """
        if not self.model_ic:
            # get initial condition from data
            xid = np.argmin(np.abs(self.x - x0))
            self.f0 = self.f[xid,...]
        else:
            """
            Initial condition for the wake model

            Args: 
                self (CurledWake)
                y (np.array): lateral axis
                z (np.array): vertical axis
                ud (float): disk velocity
                smooth_fact (float): Gaussian convolution standard deviation, 
                    equal to smooth_fact * self.dy. Defaults to 1.5. 
            """
            ### TODO get rid of hard coded values
            ct = 0.75
            yaw = 0
            yG, zG = np.meshgrid(self.y, self.z, indexing='ij')
            kernel_y = np.arange(-10, 11)[:, None] * self.dy
            kernel_z = np.arange(-10, 11)[None, :] * self.dz

            turb = (yG**2 + zG**2) < (0.5)**2
            # gauss = np.exp(-(yG**2 + zG**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
            gauss = np.exp(-(kernel_y**2 + kernel_z**2) / (np.sqrt(self.dy * self.dz) * smooth_fact)**2 / 2)
            gauss /= np.sum(gauss)
            a = 0.5 * (1 - np.sqrt(1 - ct * np.cos(yaw)**2))
            delta_u = -2 * a * self.get_ud()

            self.f0 = convolve2d(turb, gauss, 'same') * delta_u
    
    def get_ud(self, weighting='disk')-> float: 
        """
        Gets background disk velocity by numerically integrating self.U
        """
        if self.ub.ndim == 1: 
            r = 0.5
            zids = abs(self.zg) < r  # within the rotor area
            if weighting == 'disk': 
                # weight based on the area of the "disk"
                A = np.trapz(np.sqrt(r**2 - self.z[zids]**2)) 
                return np.trapz(self.ub[zids] * np.sqrt(r**2 - self.z[zids]**2)) / A
            
            elif weighting == 'equal': 
                return np.mean(self.ub[zids])
            elif weighting == 'hub': 
                return np.interp(0, self.z, self.ub)  # hub height velocity
            else: 
                raise NotImplementedError('get_ud(): `weighting` must be "disk", "equal", or "hub". ')
        else: 
            xG, yG, zG = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # creates 3D meshgrid
            xT, yT, zT = (0, 0, 0)  # coordinates of turbine
            R = 0.5
            kernel = ((yG - yT)**2 + (zG - zT)**2 < R**2)
            kernel_normalized = kernel / np.sum(kernel)

            return np.sum(np.mean(self.ub, axis=0) * kernel_normalized)

    def dfdx(self, x, f, nuT=None): 
            """du/dt function for delta u"""
            xid = np.argmin(np.abs(self.x - x))
            u = self.u[xid,...] + f

            dfdy = finite_diff(f.T, self.dy).T

            # wall BC
            f_new = top_bc(wall_bc(f))
            dfdz = finite_diff(f_new, self.dz)[...,1:-1]


            if nuT is None and self.turbulence_model is not None:
                nuT = self.nuT[xid,...]

            rhs = 0
            for key, val in self.rhs_terms.items():
                if 'model' in key:
                    if self.turbulence_model == 'scott_nonlinear':
                        dUbdy = finite_diff(self.u[xid,...].T, self.dy).T  
                        dUbdz = finite_diff(self.u[xid,...], self.dz)
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff(nuT* (dfdy + dUbdy).T, self.dy).T \
                            + finite_diff(nuT * (dfdz + dUbdz), self.dz) \
                            + finite_diff((self.nuT_base[xid,...] * dfdy).T, self.dy).T \
                            + finite_diff(self.nuT_base[xid,...] * dfdz, self.dz)
                        
                    elif self.turbulence_model == 'scott_nuT_y': 
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT_y[xid,...] * (dfdy)).T, self.dy).T \
                            + finite_diff(self.nuT[xid,...] * (dfdz), self.dz)
                        
                    elif self.turbulence_model == 'scott_nuT_y_only':
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((self.nuT_y[xid,...] * (dfdy)).T, self.dy).T

                    elif self.turbulence_model == 'scott_nuT_z_only':
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff(nuT * (dfdz), self.dz)

                    else:
                        self.rhs_terms['xturb_model'][xid,...] = finite_diff((nuT * (dfdy)).T, self.dy).T \
                                + finite_diff(nuT * (dfdz), self.dz)

                rhs += val[xid,...]
                # rhs += val[xid,...]

            # dfdz = self.dfdz[xid,...]
            # dfdy = self.dfdy[xid,...]
            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/u
    
    def _compute_vw(self)-> None: 
        """
        Use Lamb-Oseen vortices to compute curling
        """
        if self.LES_delta_v_delta_w_adv:
            return
        if self.yaw == 0: 
            return

        ct = 0.75 #TODO this is hardcoded
        sigma = 0.2
        u_inf = self.get_ud('hub')
        r_i = np.linspace(-(self.d - self.dz) / 2, (self.d - self.dz) / 2, self.N) 

        print(u_inf)

        Gamma_0 = 0.5 * self.d * u_inf * ct * np.sin(self.yaw) * np.cos(self.yaw)**2
        Gamma_i = Gamma_0 * 4 * r_i / (0.5 * np.sqrt(0.5**2 - (r_i / self.d)**2))

        # generally, vortices can decay. So sigma should be a vector
        sigma = sigma * self.d * np.ones_like(self.x) 
        
        # now we build the main summation, which is 4D (x, y, z, i)
        # yg_total = np.arange(self.y[0]-self.y[-1], self.y[-1]+self.dy, self.dy)
        # zg_total = np.arange(self.z[0]-self.z[-1], self.z[-1]+self.dz, self.dz)
        # yG, zG = np.meshgrid(yg_total, zg_total, indexing='ij')
        yG, zG = np.meshgrid(self.y, self.z, indexing='ij')
        yG = yG[None, ..., None]
        zG = zG[None, ..., None]
        r4D = yG**2 + (zG - r_i[None, None, None, :])**2  # 4D grid variable

        # mask for the ground effect
        # ind_y = np.argmin(np.abs(yg_total - self.yg[0]))
        # ind_z = np.argmin(np.abs(zg_total - self.zg[0]))
        # mask = np.ones(np.shape(r4D))
        # mask[:,:,:ind_y,:ind_z] = -1

        # put pieces together: 
        exponent = 1 - np.exp(-r4D / sigma[..., None, None, None]**2)
        summation = exponent / (2 * np.pi * r4D) * Gamma_i[None, None, None, :]

        v = np.sum(summation * (zG - r_i[None, None, None, :]), axis=-1)  # sum all vortices
        w = np.sum(summation * -yG, axis=-1)
        self.dv = -v * (self.x >= 0)[:, None, None]
        self.dw = -w * (self.x >= 0)[:, None, None]


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
                 Lref: float = 1.,
                 model_ic: bool = False):
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
        self.model_ic = model_ic

        super().__init__(io, padeops, xlim, ylim, zlim, dx, dy, dz, Uref, Lref)         

    def get_ic(self, x0):
        """
        Returns the initial condition from the x0 location
        """
        if not self.model_ic:
            xid = np.argmin(np.abs(self.x - x0))
            self.f0 = self.f[xid,...]
        else:
            self.f0 = np.zeros(np.shape(self.f[0,...]))

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
            if 'TKE_shear_production_wake' not in self.io.budget:
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
            self.du = self.io.budget['delta_u'][xid,yid,zid]/self.Uref
            self.ub = self.base_io.budget['ubar'][xid,yid,zid]/self.Uref
            if self.LES_delta_v_delta_w_adv:
                self.v = (self.base_io.budget['vbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'] + self.io.budget['delta_w'])[xid,yid,zid]/self.Uref

                self.dv = self.io.budget['delta_v'][xid,yid,zid]/self.Uref
                self.vb = self.base_io.budget['vbar'][xid,yid,zid]/self.Uref
                self.dw = self.io.budget['delta_w'][xid,yid,zid]/self.Uref
                self.wb = self.base_io.budget['wbar'][xid,yid,zid]/self.Uref
            else:
                self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref

                self.dv = np.zeros(np.shape(self.io.budget['delta_v'][xid,yid,zid]))
                self.vb = self.base_io.budget['vbar'][xid,yid,zid]/self.Uref
                self.dw = np.zeros(np.shape(self.io.budget['delta_v'][xid,yid,zid]))
                self.wb = self.base_io.budget['wbar'][xid,yid,zid]/self.Uref

            # #TEST
            # self.dfdz = np.gradient(self.f, self.dz, axis=2)
            # self.dfdy = np.gradient(self.f, self.dy, axis=1)






