"""
Turbulence model object for forward marching 

Kerry Klemmer
September 2024
"""
import numpy as np
from padeopsIO import budget_utils
from postprocess.utils import compute_ambient_width, compute_fwhm



class MixingLengthModel():
    def __init__(self):
        pass

    def calculate_ell(self, **kwargs):
        pass

class constant_MLModel(MixingLengthModel):
    """
    contant mixing length
    """   
    def __init__(self, constant: float = 0.08):
        self.ell = constant


class wake_width_MLModel(MixingLengthModel):
    """
    contant mixing length
    """   
    def __init__(self, 
                 ti_h: float = 1,
                 type: str = "ambient", 
                 ub_val: float = 1):

        self.ti_h = ti_h
        self.type = type
        self.ub_val = ub_val

        self.ell_arr = []
        

    def calculate_ell(self, du, z, y, xid, **kwargs):

        zh = np.argmin(np.abs(z))

        if self.type == 'ambient':
            [ind1, ind2] = compute_ambient_width(du[:,zh], val=-0.05*self.ub_val[xid])
            ell = y[ind2] - y[ind1]

            if ell == 0:
                ell_c = 0.08
            else:
                ell_c = ell

        self.ell = ell_c

        self.ell_arr.append(ell)


