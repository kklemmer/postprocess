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
                 type: str = "ambient"):

        self.ti_h = ti_h
        self.type = type

        self.ell_arr = []
        

    def calculate_ell(self, du, z, y, **kwargs):

        if self.type == 'ambient':
            [ind1, ind2] = compute_ambient_width(du, frac=-0.05)
            print(ind1, ind2)
            ell = y[ind2] - y[ind1]

            if ell == 0:
                ell = 0.08
            else:
                ell = 0.75 * self.ti_h * ell

        self.ell = ell

        self.ell_arr.append(ell)


