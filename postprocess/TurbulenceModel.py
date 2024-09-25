"""
Turbulence model object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np
from typing import Optional
from scipy import ndimage
from padeopsIO import budget_utils
from postprocess.utils import finite_diff, wall_bc, top_bc
from postprocess.Physics import Physics
from postprocess.Physics import WakeTKE
from postprocess import MixingLengthModel

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

class kl_LES_TurbulenceModel(TurbulenceModel):
    def __init__(self, io,
                padeops: bool=True, 
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
                ell = 0.16 * 0.5):

        super().__init__(io=io, padeops=padeops, base_io=base_io, prim_io=prim_io, xlim=xlim, ylim=ylim, zlim=zlim, dx=dx, dy=dy, dz=dz, Uref=Uref, Lref=Lref) 

        if isinstance(ell, float):
            self.ell = ell
        else:
            self.ell = ell[self.xid, np.newaxis, np.newaxis]

        self.calculate_nuT()
        self.integrate = False

    def calculate_nuT(self):
        if "TKE_wake" not in self.io.budget:
            if "TKE" not in self.base_io.budget:
                self.base_io.read_budgets(['uu', 'vv', 'ww'])
                self.base_io.budget['TKE'] = 0.5 * (self.base_io.budget['uu'] + self.base_io.budget['vv'] + self.base_io.budget['ww'])
            if "TKE" not in self.prim_io.budget:
                self.prim_io.read_budgets(['uu', 'vv', 'ww'])
                self.prim_io.budget['TKE'] = 0.5 * (self.prim_io.budget['uu'] + self.prim_io.budget['vv'] + self.prim_io.budget['ww'])

            self.io.budget['TKE_wake'] = self.prim_io.budget['TKE'] - self.base_io.budget['TKE']

        self.tke = self.io.budget['TKE_wake'][self.xid, self.yid, self.zid] / self.Uref**2
        self.base_tke = self.base_io.budget['TKE'][self.xid, self.yid, self.zid] / self.Uref**2

        self.nuT = np.sqrt(self.tke + self.base_tke) * self.ell



class kl_exact_TurbulenceModel(WakeTKE, TurbulenceModel):
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
        # self.ell = self.nuT_exact / np.sqrt(self.f)
        # self.ell[np.isnan(self.ell)] = 0

        self.ell = 1

        self.tke = np.zeros(np.shape(self.f))
        self.tke[0,...] = self.f0

        self.nuT = np.zeros(np.shape(self.f))
        self.nuT[0,...] = self.nuT_exact[0,...]

        self.integrate = False

class kl_TurbulenceModel(WakeTKE, kl_LES_TurbulenceModel):
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
                 Lref: float = 1., 
                 ell_model_str: str = "",
                 ell_LES: bool = True,
                 c_trans: float = 1.,
                 c_diss: float = 1.,
                 c_nu: float = 1.,
                 ml_type: str = "ambient",
                 model_ic: bool = False,
                 x0_nw: float = 0):
        """
        Calls the constructor of Physics
        """
        self.LES_prod = LES_prod
        if LES_prod:
            self.prod_bool = 0
        else:
            self.prod_bool = 1

        self.LES_buoy = LES_buoy
        if LES_buoy:
            self.buoy_bool = 0
        else:
            self.buoy_bool = 1

        self.LES_diss = LES_diss
        if LES_diss:
            self.diss_bool = 0
        else:
            self.diss_bool = 1

        self.LES_turb_trans = LES_turb_trans
        if LES_turb_trans:
            self.trans_bool = 0
        else:
            self.trans_bool = 1

        self.LES_pres_trans = LES_pres_trans 
        self.LES_sgs_trans = LES_sgs_trans
        self.LES_delta_v_delta_w_adv = LES_delta_v_delta_w_adv

        self.c_trans = c_trans
        self.c_diss = c_diss
    
        self.x0_nw = x0_nw

        super().__init__(io=io, padeops=padeops, LES_prod=LES_prod, LES_buoy=LES_buoy, LES_diss=LES_diss, LES_turb_trans=LES_turb_trans,
                 LES_pres_trans=LES_pres_trans, LES_sgs_trans=LES_sgs_trans, LES_delta_v_delta_w_adv=LES_delta_v_delta_w_adv,
                 base_io=base_io, prim_io=prim_io, xlim=xlim, ylim=ylim, zlim=zlim, dx=dx, dy=dy, dz=dz, Uref=Uref, Lref=Lref, model_ic=model_ic) 

        # initialize mixing length array
        self.ell = np.zeros([self.nx, self.ny, self.nz])
        self.c_nu = c_nu

        if "TKE_wake" not in self.io.budget:
            if "TKE" not in self.base_io.budget:
                self.base_io.read_budgets(['uu', 'vv', 'ww'])
                self.base_io.budget['TKE'] = 0.5 * (self.base_io.budget['uu'] + self.base_io.budget['vv'] + self.base_io.budget['ww'])
            if "TKE" not in self.prim_io.budget:
                self.prim_io.read_budgets(['uu', 'vv', 'ww'])
                self.prim_io.budget['TKE'] = 0.5 * (self.prim_io.budget['uu'] + self.prim_io.budget['vv'] + self.prim_io.budget['ww'])

            self.io.budget['TKE_wake'] = self.prim_io.budget['TKE'] - self.base_io.budget['TKE']

        self.tke = self.io.budget['TKE_wake'][self.xid, self.yid, self.zid] / self.Uref**2
        self.base_tke = self.base_io.budget['TKE'][self.xid, self.yid, self.zid] / self.Uref**2


        # initialize mixing length model
        if ell_model_str == "wake_width":
            zh = np.argmin(np.abs(self.z))
            self.ell_model = MixingLengthModel.wake_width_MLModel(ti_h=np.mean((np.sqrt(self.base_tke)/self.ub)[...,np.argmin(np.abs(self.z))], axis=(0,1)), 
                                                                  type=ml_type, ub_val = np.mean(self.base_io.budget['ubar'][...,zh], axis=1)/self.Uref)
            self.ell_model.calculate_ell(du=self.du[0,...], z=self.z, y=self.y, xid=0)
            self.ell[0,:,:] = self.ell_model.ell * np.ones([self.ny, self.nz])
        else:
            self.ell_model = MixingLengthModel.constant_MLModel()
            self.ell_model.calculate_ell()
            self.ell = self.ell_model.ell * np.ones([self.nx, self.ny, self.nz])

        self.calculate_nuT()

        self.nuT[0,...] = self.nuT[0,...] * 0.5

        self.integrate = True

    def calculate_nuT(self, x=None, k=None, du=None, return_nuT=False):
        if k is None:
            self.nuT = self.c_nu * np.sqrt(self.tke + self.base_tke) * self.ell
        else:
            xloc = np.argmin(np.abs(self.x - x))

            if x <= self.x0_nw:
                self.ell_model.calculate_ell(du=du, z=self.z, y=self.y, xid=xloc)

                # if np.abs(self.ell_model.ell - ell_old)/ell_old > 0.05 :
                #     self.ell_model.ell = ell_old

                self.ell[xloc,:,:] = self.ell_model.ell * np.ones([self.ny, self.nz])       


                nuT = self.c_nu * np.sqrt(self.base_tke[xloc,...]) * self.ell[xloc,...] * 0.5

                if return_nuT:
                    
                    return nuT
                else:
                    self.nuT[xloc,...] = nuT

                return
            else:
                if du is not None:
                    ell_old = self.ell_model.ell
                    self.ell_model.calculate_ell(du=du, z=self.z, y=self.y, xid=xloc)

                    # if np.abs(self.ell_model.ell - ell_old)/ell_old > 0.05 :
                    #     self.ell_model.ell = ell_old

                    self.ell[xloc,:,:] = self.ell_model.ell * np.ones([self.ny, self.nz])       

                tke = k + self.base_tke[xloc,...]
                tke[tke < 0] = 0

                nuT = self.c_nu * np.sqrt(tke) * self.ell[xloc,...]
                # nuT[nuT == np.nan] = 0
                if return_nuT:
                    return nuT
                else:
                    self.nuT[xloc,...] = nuT



    def dfdx(self, x, f, du=None, nuT=None): 
            """du/dt function"""

            xid = np.argmin(np.abs(self.x - x))

            if du is None:
                u = self.u[xid,...]
                du = self.du[xid,...]
            else:
                u = du + self.ub[xid,...]
            if nuT is None:
                self.calculate_nuT()
                nuT = self.nuT[xid,...]

            dfdy = finite_diff(f.T, self.dy).T
            
            # wall BC
            f_new = top_bc(wall_bc(f))
            dfdz = finite_diff(f_new, self.dz)[...,1:-1]


            ddudy = finite_diff(du.T, self.dy).T

            # wall BC
            du_new = wall_bc(du)
            ddudz = finite_diff(du_new, self.dz)[...,:-1]

            dubdy = finite_diff(self.ub[xid,...].T, self.dy).T

            # wall BC
            ub_new = wall_bc(self.ub[xid,...])
            dubdz = finite_diff(ub_new, self.dz)[...,:-1]
            
            dudy = ddudy + dubdy
            dudz = ddudz + dubdz

            rhs = 0
            # add terms from LES (if any)
            for key, val in self.rhs_terms.items():
                rhs += val[xid,...]

            # # production           
            rhs += self.prod_bool * (nuT * (ddudy*dudy + ddudz*dudz)) #+ 2*(ddvdy + 2*dvbdy)*ddvdy + 2*(ddwdz + 2*dwbdz)*ddwdz\
                                    #  + (ddvdz + 2*dvbdz)*ddvdz + 2 * (ddvdz + 2*dvbdz)*ddwdy + (ddwdy + 2*dwbdy)*ddwdy)

            # self.rhs_terms['prod'][xid,...] = self.prod_bool * (np.sqrt(self.base_tke)/self.ub)[xid,...] * (nuT * (ddudy**2 + ddudz**2 + ddvdy**2 + ddwdz**2 \
            #                                + ddudy * dubdy + ddudz * dubdz + ddvdy * dvbdy + ddwdz * dwbdz))
            # turbulent transport
            rhs += self.trans_bool * self.c_trans * (finite_diff((nuT * dfdy).T, self.dy).T \
                    + finite_diff(nuT * dfdz, self.dz))
            
            # # dissipation
            tke = f #+ self.base_tke[xid,...]
            tke[tke < 0] = 0
            diss = - self.diss_bool * self.c_diss * tke**(3/2) /self.ell[xid,...]
            diss[self.diss_bool == 0] = 0
            rhs += diss

            # turbine sink term
            if x == 0:
                ct = 0.75
                a = 0.25
                ctp = ct/(1-a)**2

                xG, yG, zG = np.meshgrid(self.x, self.y, self.z, indexing='ij')  # creates 3D meshgrid
                xT, yT, zT = (0, 0, 0)  # coordinates of turbine
                R = 0.5
                kernel = ((yG - yT)**2 + (zG - zT)**2 < R**2)
                kernel_normalized = kernel / np.sum(kernel)

                kd = np.sum(np.mean(self.base_tke, axis=0) * kernel_normalized)
                ud = np.sum(np.mean(self.ub, axis=0) * kernel_normalized)

                Sk = - 0.5 * ctp * np.pi/4 * (4/3 * kd * ud * (2/3 * kd)**(3/2))



                rhs += Sk * kernel_normalized[xid,...]

            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/u


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
            x_data = 2 * self.io.budget['S13'] * self.prim_io.stream_mask
            y_data = self.io.budget['uw_wake'] * self.prim_io.stream_mask
        else:
            x_data = 2 * self.io.budget['S13'][self.xid, self.yid, self.zid]
            y_data = self.io.budget['uw_wake'][self.xid, self.yid, self.zid]
            
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT = self.Cz * slopes / self.Uref
        self.nuT[self.nuT < 0] = np.min(self.nuT)
        self.nuT_err = self.Cz * slope_err / self.Uref

        return

class scott_model_TurbulenceModel(TurbulenceModel):
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
                 Lref: float = 1.):
        
        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref) 

        self.calculate_nuT()
        self.integrate = False

    def calculate_nuT(self):
        """
        Calculate the 1D nuT following the model in Scott et al 2023
        """
        Ub = np.mean(self.base_io.budget['ubar'][...,14])/self.Uref
        Ct = 0.75
        A = 0.5 * Ub * np.sqrt(1-Ct)/2
        sigma = 5.5

        self.nuT = np.broadcast_to(A*(0.01 + np.multiply(self.x/sigma**2, np.exp(-self.x**2/(2*sigma**2))))[:,np.newaxis, np.newaxis], (self.nx, self.ny, self.nz))
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
        x_data = 2 * self.base_io.budget['S13'][self.xid, self.yid, self.zid]
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
            x_data = 2 * self.io.budget['S13'] * self.prim_io.stream_mask
            y_data = self.io.budget['delta_uw'] * self.prim_io.stream_mask
        else:
            x_data = 2 * self.io.budget['S13'][self.xid, self.yid, self.zid]
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
            x_data = 2 * self.io.budget['S12'] * self.prim_io.stream_mask
            y_data = self.io.budget['uv_wake'] * self.prim_io.stream_mask
        else:
            x_data = 2 * self.io.budget['S12'][self.xid, self.yid, self.zid]
            y_data = self.io.budget['uv_wake'][self.xid, self.yid, self.zid]
            
        slopes, slope_err = calculate_slope(x_data, y_data)

        self.nuT_y = self.Cy * slopes / self.Uref
        self.nuT_y_err = self.Cy * slope_err / self.Uref

        return

class blackadar_TurbulenceModel(TurbulenceModel):
    """
    Eddy viscosity from Blackadar 1962
    """

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
                Lref: float = 1.,
                fc: float = 27):

        super().__init__(io, padeops, base_io, prim_io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref)
        self.fc = fc / self.Lref

        self.calculate_nuT()
        self.integrate = False

    def calculate_nuT(self):
        z = self.base_io.zLine[self.zid] # ensures the correct mixing length where 0 is at the wall
        kappa = 0.4
        lam = 27 / 126
        ell = kappa * z/(1 + kappa * z/lam)

        self.nuT = 0.01 + np.broadcast_to((4*ell**2 * (1/self.Uref) * np.abs(np.mean(np.gradient(self.base_io.budget['ubar'][self.xid,self.yid,self.zid], 
                                                                         self.dz, axis=2), axis=(0,1))))[np.newaxis, np.newaxis,:], (self.nx, self.ny, self.nz))

def calculate_slope(x_data, y_data):

    slopes = np.zeros(np.shape(x_data)[0])
    slope_err = np.zeros(np.shape(x_data)[0])

    for i in range(len(slopes)):
        slope, intercept, r_value, p_value, std_err = linregress(-x_data[i].ravel(), y_data[i].ravel())
        slopes[i] = slope
        slope_err[i] = std_err

    return slopes, slope_err