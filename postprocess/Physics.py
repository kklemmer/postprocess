"""
Integration object for forward marching 

Kerry Klemmer
June 2024
"""
import numpy as np
from typing import Optional

class Physics():

    def __init__(self, 
                padeops: bool=False, 
                io: Optional[object] = None,
                xlim: Optional[list] = None,
                ylim: Optional[list] = None,
                zlim: Optional[list] = None,
                dx: float = 0.1,
                dy: float = 0.1,
                dz: float = 0.1,
                Uref: float = 1.,
                Lref: float = 1.):

        self.padeops = padeops
        self.io = io
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

        print(self.dx)

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
            u = self.u[xid,...] + f

            rhs = 0
            for key, val in self.rhs_terms.items():
                print(key)
                rhs += val[xid,...]

            dfdy = np.gradient(f, self.dy, axis=0)
            dfdz = np.gradient(f, self.dz, axis=1)

            return (rhs - self.v[xid,...] * dfdy - self.w[xid,...] * dfdz)/u
        

class WakeDeficitX(Physics):
    """
    Class that extends Physics to model streamwise wake deficit
    """
    
    def __init__(self, padeops, io, 
                 LES_dpdx: bool = True,
                 LES_delta_v_delta_w_adv: bool = True,
                 LES_xturb: bool = True,
                 LES_xCor: bool = True,
                 LES_xSGS: bool = True, 
                 base_io:Optional[object] = None,               
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
        self.LES_dpdx = LES_dpdx
        self.LES_delta_v_delta_w_adv = LES_delta_v_delta_w_adv
        self.LES_xturb = LES_xturb
        self.LES_xCor = LES_xCor
        self.LES_xSGS = LES_xSGS  

        super().__init__(padeops, io, xlim, ylim, zlim, dx, dy, dz, Uref, Lref)         

    def create_terms_dict(self):
        """
        Create the necessary dictionary of rhs terms
        """
        self.rhs_terms = {}

        if self.padeops:
            budget_terms = ['delta_u', 'delta_v', 'delta_w', 'dpdx', 'xAdv_base_delta_fluc', 
                'xAdv_delta_delta_fluc', 'xAdv_delta_base_fluc', 'xCor', 'xAdv_delta_base_mean', 'xSGS']
            
            self.io.read_budgets(budget_terms)
            self.base_io.read_budgets(['ubar', 'vbar', 'wbar'])

            xid = self.xid  
            yid = self.yid
            zid = self.zid

            nonDim = self.Lref/(self.Uref**2)
            print(self.Lref)

            if self.LES_dpdx:
                self.rhs_terms['dpdx'] = self.io.budget['dpdx'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['dpdx'] = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))

            if self.LES_delta_v_delta_w_adv:
                self.rhs_terms['delta_v_delta_w_adv'] = (self.io.budget['xAdv_delta_base_mean'][xid,yid,zid]) * nonDim
            else:
                self.rhs_terms['delta_v_delta_w_adv'] = np.zeros(np.shape(self.io.budget['xAdv_delta_base_mean'][xid,yid,zid]))

            if self.LES_xturb:
                self.rhs_terms['xturb'] = (self.io.budget['xAdv_delta_delta_fluc'][xid,yid,zid] + self.io.budget['xAdv_delta_base_fluc'][xid,yid,zid] + self.io.budget['xAdv_base_delta_fluc'][xid,yid,zid]) * nonDim
            else:
                self.rhs_terms['xturb'] = np.zeros(np.shape(self.io.budget['dpdx'][xid,yid,zid]))

            if self.LES_xCor:
                self.rhs_terms['xCor'] = self.io.budget['xCor'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['xCor'] = np.zeros(np.shape(self.io.budget['xCor'][xid,yid,zid]))
            
            if self.LES_xSGS:
                self.rhs_terms['xSGS'] = self.io.budget['xSGS'][xid,yid,zid] * nonDim
            else:
                self.rhs_terms['xSGS'] = np.zeros(np.shape(self.io.budget['xSGS'][xid,yid,zid]))

            self.f = self.io.budget['delta_u'][xid,yid,zid]/self.Uref

            self.u = (self.base_io.budget['ubar'])[xid,yid,zid]/self.Uref
            if self.LES_delta_v_delta_w_adv:
                self.v = (self.base_io.budget['vbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'] + self.io.budget['delta_v'])[xid,yid,zid]/self.Uref
            else:
                self.v = (self.base_io.budget['vbar'])[xid,yid,zid]/self.Uref
                self.w = (self.base_io.budget['wbar'])[xid,yid,zid]/self.Uref