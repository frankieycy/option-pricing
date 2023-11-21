import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from american import *
plt.switch_backend('Agg')

FLAT_VOL        = {'sig': 0.2}
SVI_PARAMS_FLAT = {'v0': 0.04, 'v1': 0.04, 'v2':0.04, 'k1': 1, 'k2': 2,   'rho': 0,    'eta': 0, 'gam': 0  }
SVI_PARAMS_SPX  = {'v0': 0.09, 'v1': 0.04, 'v2':0.04, 'k1': 5, 'k2': 0.1, 'rho': -0.5, 'eta': 1, 'gam': 0.5}

def test_SviPowerLaw():
    k = np.arange(-1,1.05,0.05)
    T = np.arange(0.1,1.05,0.05)
    kk,TT = np.meshgrid(k,T)
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    sigI = svi.IVolFunc(kk,TT)
    sigL = svi.LVolFunc(kk,TT)
    sigI = VolSurfaceMatrixToDataFrame(sigI,k,T)
    sigL = VolSurfaceMatrixToDataFrame(sigL,k,T)
    print('---sigI---')
    print(sigI)
    print('---sigL---')
    print(sigL)
    PlotImpliedVol(sigI,sigL,xlim=(-1,1),ylim=(12,100),figname='test/impliedvol.png')

def test_LatticePricer():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'C'
    ex = 'E'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,'crank-nicolson')
    L = LatticePricer(S)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    print(O)
    print(S)
    print(C)
    print(L)
    print('---C.rangeS---')
    print(C.rangeS)
    print('---O.exBdryLV---')
    print(O.exBdryLV)
    print('---O.pxGridLV---')
    print(O.pxGridLV)
    if ex == 'E':
        D = np.exp(-r*T)
        F = S.ForwardFunc(T)
        V = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
        sig = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
        print('---BSPrice---')
        print('pxTrue',V,'pxLatt',O.px)
        print('sigTrue',O.ivLV,'sigLatt',sig)

def test_DeAmericanize():
    # sigI and sigA should converge for nX and nT large enough
    # Implicit scheme requires nT~1000
    # Can Crank-Nicolson achieve accuracy with nT~200?
    svi = SviPowerLaw(**SVI_PARAMS_FLAT)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 200
    nT = 200
    x0 = -2
    x1 = 2
    pc = 'P'
    ex = 'E'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,'implicit')
    L = LatticePricer(S)
    F = S.ForwardFunc(T)
    k = np.log(K/F)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    L.DeAmericanize(O,C)
    print(O)
    print(S)
    print(C)
    print(L)
    print('---O.exBdryLV---')
    print(O.exBdryLV)
    print('---O.exBdryFV---')
    print(O.exBdryFV)
    print('---svi.IVol---')
    print(svi.IVolFunc(k,T))
    print('---svi.LVol---')
    print(svi.LVolFunc(k,T))
    if ex == 'E':
        D = np.exp(-r*T)
        sig = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
        print('---BSVol---')
        print('sigTrue',O.ivLV,'sigDeAm',O.ivFV,'sigLatt',sig)

def test_AmericanVolSurface():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 200
    nT = 200
    G  = 6
    S  = Spot(S0,r,0,svi)
    C  = LatticeConfig(S0,'implicit')
    A  = AmericanVolSurface(S,C,nX,nT,G)
    k  = np.arange(-0.5,0.6,0.1)
    sigI = svi.IVolFunc(k,T)
    sigA = A.IVolFunc(k,T)
    print('---sigI---')
    print(pd.Series(sigI,index=k))
    print('---sigA---')
    print(pd.Series(sigA,index=k))
    print('---A.log---')
    print(A.log)

def test_LatticePricerAccuracy_FlatVol():
    svi = SviPowerLaw(**SVI_PARAMS_FLAT)
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    G  = 5
    ex = 'E'
    kk = np.arange(-0.5,0.55,0.05)
    TT = np.arange(0.1,1.1,0.1)
    m  = 'crank-nicolson'
    b  = 'gamma'
    S  = Spot(S0,r,0,svi)
    C  = LatticeConfig(S0,m,b)
    L  = LatticePricer(S)
    with open(f'test/lattice_eu_acc_flatvol_nX={nX}_nT={nT}_G={G}_dk={round(kk[1]-kk[0],2)}_m={m}_b={b}.csv','w') as f:
        f.write(f'#S0={S0},r={r},nX={nX},nT={nT},G={G},m={m},vs={svi}\n')
        f.write('k,K,T,ex,pc,pxTrue,pxLatt,sigTrue,sigLatt,sigLV,sigErr\n')
        for T in TT:
            T = round(T,2)
            print(f'Running lattice at T={T} ...')
            D = np.exp(-r*T)
            F = S.ForwardFunc(T)
            for k in tqdm(kk):
                sig0 = svi.LVolFunc(k,T)
                x0 = -G*sig0*np.sqrt(T)
                x1 = G*sig0*np.sqrt(T)
                K  = F*np.exp(k)
                pc = 'P' if k<=0 else 'C'
                O  = Option(K,T,pc,ex)
                C.initGrid(nX,nT,[x0,x1],[0,T],K)
                L.SolveLattice(O,C)
                pxTrue  = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
                pxLatt  = O.px
                sigTrue = O.ivLV
                sigLatt = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
                sigLV   = O.lvLV
                sigErr  = sigLatt-sigTrue
                f.write(f'{k},{K},{T},{ex},{pc},{pxTrue},{pxLatt},{sigTrue},{sigLatt},{sigLV},{sigErr}\n')

def test_LatticePricerAccuracy_SpxVol():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    G  = 5
    ex = 'E'
    kk = np.arange(-0.5,0.52,0.02)
    TT = np.arange(0.1,1.1,0.1)
    m  = 'crank-nicolson'
    b  = 'gamma'
    S  = Spot(S0,r,0,svi)
    C  = LatticeConfig(S0,m,b)
    L  = LatticePricer(S)
    with open(f'test/lattice_eu_acc_spxvol_nX={nX}_nT={nT}_G={G}_dk={round(kk[1]-kk[0],2)}_m={m}_b={b}.csv','w') as f:
        f.write(f'#S0={S0},r={r},nX={nX},nT={nT},G={G},m={m},vs={svi}\n')
        f.write('k,K,T,ex,pc,pxTrue,pxLatt,sigTrue,sigLatt,sigLV,sigErr\n')
        for T in TT:
            T = round(T,2)
            print(f'Running lattice at T={T} ...')
            D = np.exp(-r*T)
            F = S.ForwardFunc(T)
            for k in tqdm(kk):
                sig0 = svi.LVolFunc(k,T)
                x0 = -G*sig0*np.sqrt(T)
                x1 = G*sig0*np.sqrt(T)
                K  = F*np.exp(k)
                pc = 'P' if k<=0 else 'C'
                O  = Option(K,T,pc,ex)
                C.initGrid(nX,nT,[x0,x1],[0,T],K)
                L.SolveLattice(O,C)
                pxTrue  = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
                pxLatt  = O.px
                sigTrue = O.ivLV
                sigLatt = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
                sigLV   = O.lvLV
                sigErr  = sigLatt-sigTrue
                f.write(f'{k},{K},{T},{ex},{pc},{pxTrue},{pxLatt},{sigTrue},{sigLatt},{sigLV},{sigErr}\n')

def test_LatticePricer_ATMEuPut():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'P'
    ex = 'E'
    m  = 'crank-nicolson'
    b  = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,m,b)
    L = LatticePricer(S)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    pxGrid = O.pxGridLV
    exBdry = O.exBdryLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/atm_eu_put_pxgrid_m={m}_b={b}.csv')
    exBdry.to_csv(f'test/atm_eu_put_exbdry_m={m}_b={b}.csv')

def test_LatticePricer_ATMEuCall():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0
    q  = 0.05
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'C'
    ex = 'E'
    m  = 'crank-nicolson'
    b  = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,q,svi)
    C = LatticeConfig(S0,m,b)
    L = LatticePricer(S)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    pxGrid = O.pxGridLV
    exBdry = O.exBdryLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/atm_eu_call_pxgrid_m={m}_b={b}.csv')
    exBdry.to_csv(f'test/atm_eu_call_exbdry_m={m}_b={b}.csv')

def test_LatticePricer_ATMAmPut():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'P'
    ex = 'A'
    m  = 'crank-nicolson'
    b  = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,m,b)
    L = LatticePricer(S)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    pxGrid = O.pxGridLV
    exBdry = O.exBdryLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/atm_am_put_pxgrid_m={m}_b={b}.csv')
    exBdry.to_csv(f'test/atm_am_put_exbdry_m={m}_b={b}.csv')

def test_LatticePricer_ATMAmCall():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0
    q  = 0.05
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'C'
    ex = 'A'
    m  = 'crank-nicolson'
    b  = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,q,svi)
    C = LatticeConfig(S0,m,b)
    L = LatticePricer(S)
    C.initGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    pxGrid = O.pxGridLV
    exBdry = O.exBdryLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/atm_am_call_pxgrid_m={m}_b={b}.csv')
    exBdry.to_csv(f'test/atm_am_call_exbdry_m={m}_b={b}.csv')

if __name__ == '__main__':
    # test_SviPowerLaw()
    # test_LatticePricer()
    # test_DeAmericanize()
    # test_AmericanVolSurface()
    # test_LatticePricerAccuracy_FlatVol()
    # test_LatticePricerAccuracy_SpxVol()
    # test_LatticePricer_ATMEuPut()
    # test_LatticePricer_ATMEuCall()
    test_LatticePricer_ATMAmPut()
    test_LatticePricer_ATMAmCall()
