import polars as pl
import padeopsIO as pio
import numpy as np

import seaborn as sns

from coriolis import scripts
from coriolis.streamtube import Streamtube
from coriolis.budget import Budget


def streamtube(case, method='stream3', R=0.5, tidx=86000): 
    # case.read_budgets(budget_terms=['ubar', 'vbar', 'wbar'], tidx=tidx)
    stream = Streamtube(case.xLine, case.yLine, case.zLine, 
                        case.budget['ubar'], case.budget['vbar'], case.budget['wbar'],  
                        method=method)
    stream.compute_mask(R=R)
    case.stream_mask = stream.mask


def load_cases_z0_dTdt(dTdt=[0, -0.25, -0.5, -0.75, -1], z0=[1, 10, 50], streamtube=False):
    """
    Load the budgets for the stability and z0 sweeps into a polars dataframe

    INPUTS:
    dTdt (optional) - list of surface cooling rates
    z0 (optional)   - list of surface roughness lengths
    if these are not set then all cases will be read in
    """

    # create cases dicts
    base_dir = r'/work/09024/kklemmer/stampede3/data/'
    nbl_base_dir = r'stratified_pbl_concurrent/neutral/sbl_match/'
    sbl_base_dir = r'stable_pbl_concurrent/'

    cases = []


    for i in range(len(dTdt)):
        for j in range(len(z0)):
            cases.append({})

    cases = np.reshape(cases, (len(dTdt),len(z0)))

    colors = sns.color_palette('mako', len(dTdt))

    ls = [':', '-', '-.']   

    for i in range(len(dTdt)):
        for j in range(len(z0)):
            cases[i][j]['dTdt'] = dTdt[i]
            cases[i][j]['z0'] = z0[j]
            cases[i,j]['color'] = colors[i]
            cases[i,j]['ls'] = ls[j]
            if dTdt[i] == 0:
                cases[i][j]['dir'] = base_dir + nbl_base_dir + 'z0_' + str(z0[j]) + 'cm'
                cases[i][j]['Uref'] = 12
                cases[i][j]['Lref'] = 126/1000
            else:
                cases[i][j]['dir'] = base_dir + sbl_base_dir + 'dTdt_' + str(np.abs(dTdt[i])).replace('.', '') + '/z0_' + str(z0[j]) + 'cm'
                cases[i][j]['Uref'] = 1
                cases[i][j]['Lref'] = 126/400

    # create primary, base, and deficit budget objects
    for i in range(len(dTdt)):
        for j in range(len(z0)):
                cases[i][j]['prim'] = pio.BudgetIO(cases[i][j]['dir'] + '/budgets/time', padeops=True, runid=2, normalize_origin='turbine',
                                                                normalize_grid=True, Lnorm=cases[i,j]['Lref'])
                cases[i][j]['base'] = pio.BudgetIO(cases[i][j]['dir'] + '/budgets/time/precursor', padeops=True, runid=1,
                                                                normalize_grid=True, Lnorm=cases[i,j]['Lref'])
                cases[i][j]['deficit'] = pio.DeficitIO(cases[i][j]['dir'] + '/budgets/time', padeops=True, runid=2, normalize_origin='turbine',
                                                                normalize_grid=True, Lnorm=cases[i,j]['Lref'])

                # calculate hub height wind speed
                cases[i,j]['base'].read_budgets(['ubar', 'vbar', 'Tbar'])
                cases[i,j]['u_hh'] = np.mean(np.sqrt(cases[i,j]['base'].budget['ubar'][...,14]**2 \
                                                    + cases[i,j]['base'].budget['vbar'][...,14]**2))
                
    budget_terms_to_read = ['delta_u', 'delta_v', 'delta_w']#, 'xAdv_delta_base_mean',
                        # 'xAdv_delta_base_fluc', 'xAdv_delta_delta_mean', 
                        # 'xAdv_delta_delta_fluc', 'xAdv_base_delta_fluc', 'xAdv_base_delta_mean',
                        # 'xCor', 'xSGS', 'dpdx', 'xAD']
    for i,dT in enumerate(dTdt):
        for j in range(len(z0)):
            cases[i,j]['base'].read_budgets(budget_terms=['ubar', 'vbar', 'wbar'])
            cases[i,j]['deficit'].read_budgets(budget_terms=budget_terms_to_read)
            # cases[i,j]['deficit_budget'].grad_calc(Lref=cases[i,j]['Lref'])
            cases[i,j]['prim'].read_budgets(budget_terms=['ubar', 'vbar', 'wbar'])

            if streamtube:
                if dT == 0:
                    streamtube(cases[i,j]['prim'])
                else:
                    streamtube(cases[i,j]['prim'], method='integrator')
    
    cases = list(np.reshape(cases, (len(dTdt) * len(z0),)))

    df = pl.from_dicts(cases)

    return df
                
    
