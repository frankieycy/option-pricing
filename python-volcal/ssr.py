import numpy as np
import scipy as sp
import pandas as pd
from svi import FitArbFreeSimpleSVIWithSimSeed, SVIVolSurface, SVIAtmTermStructure

def BatchFitArbFreeSimpleSVI(dfs):
    # Fit Simple SVI to each slice guaranteeing no static arbitrage
    # dfs: dict of standardized options chain df labeled by dates
    fits = dict()
    for T in dfs:
        print(f'fitting arb-free simple SVI for T={T}')
        fits[T] = FitArbFreeSimpleSVIWithSimSeed(dfs[T])
    return fits

def SVIVolSurfaceStats(fits, Texp=None):
    # SVI vol surface statistics e.g. ATM vol/skew
    # dfs: dict of SVI fit labeled by dates
    # Texp: expiries to extract stats
    EPS = 1e-5
    ts = dict()
    if Texp is not None:
        k = np.array([-EPS,0,EPS])
        Texp = np.array(Texp)
    for T in fits:
        print(f'computing vol surface stats for T={T}')
        if Texp is None:
            ts[T] = SVIAtmTermStructure(fits[T])[['atm','skew']]
        else:
            surf = SVIVolSurface(fits[T])
            surf0 = surf(k,Texp)
            ts[T] = pd.DataFrame({
                'atm': surf0[:,1],
                'skew': (surf0[:,2]-surf0[:,0])/(2*EPS),
            })
            ts[T].index = Texp
    return ts
