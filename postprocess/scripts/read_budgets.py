import polars as pl
import padeopsIO as pio
import numpy as np

def read_budget0(cases):
    """
    Reads budget0 terms (u, v, w, uu, vv, ww)
    """
    terms = ['ubar', 'vbar', 'wbar', 'uu', 'vv', 'ww']
    if 'prim' in cases:
        cases['prim'].map_elements(lambda e: e.read_budgets(terms))

    if 'base' in cases:
        cases['base'].map_elements(lambda e: e.read_budgets(terms))


    terms = ['delta_u', 'delta_v', 'delta_w']
    if 'deficit' in cases:
        cases['deficit'].map_elements(lambda e: e.read_budgets(terms))

def tke_calc(cases):
    """
    Runs tke_calc from padeopsIO
    """

    ind_def = cases.get_column_index("deficit")
    ind_base = cases.get_column_index("base")
    ind_prim = cases.get_column_index("prim")

    for row in cases.iter_rows():
        pio.budget_utils.tke_calc(row[ind_base])
        pio.budget_utils.tke_calc(row[ind_prim])
        row[ind_def].budget['TKE_wake'] = row[ind_prim].budget['TKE'] - row[ind_base].budget['TKE']

def ti_calc(cases):
    """
    Runs TI_calc from padeopsIO
    """

    ind_def = cases.get_column_index("deficit")
    ind_base = cases.get_column_index("base")
    ind_prim = cases.get_column_index("prim")
    ind_uh = cases.get_column_index("u_hh")

    for row in cases.iter_rows():
        row[ind_base].budget['TI'] = pio.budget_utils.TI_calc(row[ind_base])/row[ind_uh]
        row[ind_prim].budget['TI'] = pio.budget_utils.TI_calc(row[ind_prim])/row[ind_uh]
        row[ind_def].budget['TI_wake'] = np.sqrt(row[ind_prim].budget['TI']**2 - row[ind_base].budget['TI']**2)

def streamtube_calc(cases):
    """
    Computes streamtube for each case
    """


    from coriolis.streamtube import Streamtube

    def streamtube(case, method='stream3', R=0.5, tidx=86000): 
        # case.read_budgets(budget_terms=['ubar', 'vbar', 'wbar'], tidx=tidx)
        stream = Streamtube(case.xLine, case.yLine, case.zLine, 
                            case.budget['ubar'], case.budget['vbar'], case.budget['wbar'],  
                            method=method)
        stream.compute_mask(R=R)
        case.stream_mask = stream.mask


    ind_prim = cases.get_column_index("prim")
    ind_dTdt = cases.get_column_index("dTdt")

    for row in cases.iter_rows():
        if row[ind_dTdt] == 0:
            streamtube(row[ind_prim])
        else:
            streamtube(row[ind_prim], method='integrator')

