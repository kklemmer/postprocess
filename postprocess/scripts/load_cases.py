import polars as pl
import padeopsIO as pio
import numpy as np

def load_cases_z0_dTdt():
    """
    Load the budgets for the stability and z0 sweeps into a polars dataframe
    """

    # create cases dicts
    base_dir = r'/work/09024/kklemmer/stampede3/data/'
    nbl_base_dir = r'stratified_pbl_concurrent/neutral/sbl_match/'
    sbl_base_dir = r'stable_pbl_concurrent/'

    cases = []

    dTdt = [0, -0.25, -0.5, -0.75, -1]
    z0 = [10]


    for i in range(len(dTdt)):
        for j in range(len(z0)):
            cases.append({})

    cases = np.reshape(cases, (len(dTdt),len(z0)))

    for i in range(len(dTdt)):
        for j in range(len(z0)):
            cases[i][j]['dTdt'] = dTdt[i]
            cases[i][j]['z0'] = z0[j]
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

    cases = list(np.reshape(cases, (len(dTdt) * len(z0),)))

    df = pl.from_dicts(cases)

    return df
                
    
