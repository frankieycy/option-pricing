import time
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
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'P'
    ex = 'A'
    m = 'crank-nicolson'
    b = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,0,svi)
    C = LatticeConfig(S0,m,b,fast=True)
    L = LatticePricer(S)
    F = S.ForwardFunc(T)
    k = np.log(K/F)
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
    nX = 1000
    nT = 1000
    G  = 5
    m  = 'crank-nicolson'
    b  = 'gamma'
    for r in [0,0.02,0.05,0.1]:
        print(f'Running for r={r} ...')
        S  = Spot(S0,r,0,svi)
        C  = LatticeConfig(S0,m,b,fast=True)
        A  = AmericanVolSurface(S,C,nX,nT,G)
        k  = np.arange(-0.5,0.52,0.02)
        T = np.arange(0.1,1.05,0.05)
        kk,TT = np.meshgrid(k,T)
        sigI = svi.IVolFunc(kk,TT)
        sigA = A.IVolFunc(kk,TT)
        sigI = pd.DataFrame(sigI,index=T,columns=k)
        sigA = pd.DataFrame(sigA,index=T,columns=k)
        sigI.to_csv(f'test/deAm_vol_eu_r={r}.csv')
        sigA.to_csv(f'test/deAm_vol_am_r={r}.csv')
        print('---sigI---')
        print(sigI)
        print('---sigA---')
        print(sigA)
        # print('---A.log---')
        # print(A.log)

def test_AmericanVolSurface_data():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    nX = 1000
    nT = 1000
    G  = 5
    m  = 'crank-nicolson'
    b  = 'gamma'
    S  = Spot(S0,r,0,svi)
    C  = LatticeConfig(S0,m,b,fast=True)
    A  = AmericanVolSurface(S,C,nX,nT,G)
    k  = np.arange(-0.5,0.52,0.02)
    T  = np.arange(0.1,1.05,0.05)
    kk,TT = np.meshgrid(k,T)
    sigI = svi.IVolFunc(kk,TT)
    sigA = A.IVolFunc(kk,TT)
    sigI = pd.DataFrame(sigI,index=T,columns=k)
    sigA = pd.DataFrame(sigA,index=T,columns=k)
    print('---sigI---')
    print(sigI)
    print('---sigA---')
    print(sigA)
    f1 = []
    f2 = []
    for o in A.log:
        K  = o.K
        T  = o.T
        px = o.px
        ivLV = o.ivLV
        ivFV = o.ivFV
        lvLV = o.lvLV
        exLV = o.exBdryLV
        exFV = o.exBdryFV
        t     = exLV.index.to_numpy()
        E     = exLV.values
        F     = S.ForwardFunc(T-t)
        F0    = S.ForwardFunc(T)
        kk    = np.log(K/F0)
        ee    = np.log(E/F)
        sigLV = pd.Series(svi.IVolFunc(ee,T-t)).to_dict()
        skLV  = pd.Series(svi.ISkewFunc(ee,T-t)).to_dict()
        exLV  = exLV.reset_index(drop=True).to_dict()
        exFV  = exFV.reset_index(drop=True).to_dict()
        exLV.update({'k':kk,'K':K,'T':T,'var':'Sex_LV'})
        exFV.update({'k':kk,'K':K,'T':T,'var':'Sex_FV'})
        sigLV.update({'k':kk,'K':K,'T':T,'var':'sig_LV'})
        skLV.update({'k':kk,'K':K,'T':T,'var':'sk_LV'})
        f1.append({'k':kk,'K':K,'T':T,'px':px,'ivLV':ivLV,'ivFV':ivFV,'lvLV':lvLV})
        f2.append(exLV)
        f2.append(exFV)
        f2.append(sigLV)
        f2.append(skLV)
    f1 = pd.DataFrame(f1)
    f2 = pd.DataFrame(f2)
    f2 = f2[['k','K','T','var']+list(range(nT))]
    f1.to_csv('test/deAm_data_option.csv')
    f2.to_csv('test/deAm_data_exbdry.csv')

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
                C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
                C.InitGrid(nX,nT,[x0,x1],[0,T],K)
                L.SolveLattice(O,C)
                pxTrue  = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
                pxLatt  = O.px
                sigTrue = O.ivLV
                sigLatt = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
                sigLV   = O.lvLV
                sigErr  = sigLatt-sigTrue
                f.write(f'{k},{K},{T},{ex},{pc},{pxTrue},{pxLatt},{sigTrue},{sigLatt},{sigLV},{sigErr}\n')

def test_LatticePricerAccuracy_SpxVol_finetune():
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
    C  = LatticeConfig(S0,m,b,fast=True)
    L  = LatticePricer(S)
    with open(f'test/lattice_eu_acc_spxvol_nX={nX}_nT={nT}_TTX={MIN_TTX}_LVAR={MAX_LVAR}.csv','w') as f:
        f.write(f'#S0={S0},r={r},nX={nX},nT={nT},G={G},m={m},vs={svi},MIN_TTX={MIN_TTX},MAX_LVAR={MAX_LVAR}\n')
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
                C.InitGrid(nX,nT,[x0,x1],[0,T],K)
                L.SolveLattice(O,C)
                pxTrue  = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
                pxLatt  = O.px
                sigTrue = O.ivLV
                sigLatt = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
                sigLV   = O.lvLV
                sigErr  = sigLatt-sigTrue
                f.write(f'{k},{K},{T},{ex},{pc},{pxTrue},{pxLatt},{sigTrue},{sigLatt},{sigLV},{sigErr}\n')

def test_LatticePricerAccuracy_SpxVol_deAm():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    S0 = 1
    r  = 0
    nX = 1000
    nT = 1000
    G  = 5
    ex = 'A'
    kk = np.arange(-0.5,0.52,0.02)
    TT = np.arange(0.1,1.1,0.1)
    m  = 'crank-nicolson'
    b  = 'gamma'
    S  = Spot(S0,r,0,svi)
    C  = LatticeConfig(S0,m,b,fast=True)
    L  = LatticePricer(S)
    with open(f'test/lattice_eu_acc_spxvol_nX={nX}_nT={nT}.csv','w') as f:
        f.write(f'#S0={S0},r={r},nX={nX},nT={nT},G={G},m={m},vs={svi}\n')
        f.write('k,K,T,ex,pc,pxTrue,pxLatt,sigTrue,sigLatt,sigDeAm,sigLV,sigErr,sigErrA\n')
        for T in TT:
            T = round(T,2)
            print(f'Running lattice at T={T} ...')
            D = np.exp(-r*T)
            F = S.ForwardFunc(T)
            for k in kk:
                sig0 = svi.LVolFunc(k,T)
                x0 = -G*sig0*np.sqrt(T)
                x1 = G*sig0*np.sqrt(T)
                K  = F*np.exp(k)
                pc = 'P' if k<=0 else 'C'
                O  = Option(K,T,pc,ex)
                C.InitGrid(nX,nT,[x0,x1],[0,T],K)
                L.SolveLattice(O,C)
                L.DeAmericanize(O,C)
                pxTrue  = BlackScholesPrice(O.ivLV,K,T,D,F,pc)
                pxLatt  = O.px
                sigTrue = O.ivLV
                sigLatt = BlackScholesImpliedVol(O.px,K,T,D,F,pc,O.ivLV)[0]
                sigDeAm = O.ivFV
                sigLV   = O.lvLV
                sigErr  = sigLatt-sigTrue
                sigErrA = sigDeAm-sigTrue
                print(f'Finished k={k}, T={T} ... sigErr={sigErr} sigErrA={sigErrA}')
                f.write(f'{k},{K},{T},{ex},{pc},{pxTrue},{pxLatt},{sigTrue},{sigLatt},{sigDeAm},{sigLV},{sigErr},{sigErrA}\n')

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
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
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
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
    L.SolveLattice(O,C)
    pxGrid = O.pxGridLV
    exBdry = O.exBdryLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/atm_am_call_pxgrid_m={m}_b={b}.csv')
    exBdry.to_csv(f'test/atm_am_call_exbdry_m={m}_b={b}.csv')

def test_LatticePricer_Speedup():
    svi = SviPowerLaw(**SVI_PARAMS_SPX)
    # svi = FlatVol(0.2)
    K  = 1
    T  = 1
    S0 = 1
    r  = 0.05
    q  = 0
    nX = 1000
    nT = 1000
    x0 = -2
    x1 = 2
    pc = 'P'
    ex = 'A'
    m  = 'crank-nicolson'
    b  = 'gamma'
    O = Option(K,T,pc,ex)
    S = Spot(S0,r,q,svi)
    C = LatticeConfig(S0,m,b)
    L = LatticePricer(S)
    C.InitGrid(nX,nT,[x0,x1],[0,T],K)
    t0 = time.time()
    L.SolveLattice(O,C)
    t1 = time.time()
    dt1 = t1-t0
    px1 = O.px
    pxGrid = O.pxGridLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/speed_pxgrid_reg.csv')
    C.SetFast(True)
    t0 = time.time()
    L.SolveLattice(O,C)
    t1 = time.time()
    dt2 = t1-t0
    px2 = O.px
    pxGrid = O.pxGridLV
    pxGrid.columns = C.XToS(pxGrid.columns)
    pxGrid.to_csv(f'test/speed_pxgrid_opt.csv')
    print(f'regular mode : price={px1} time={dt1}')
    print(f'fast mode    : price={px2} time={dt2}')

if __name__ == '__main__':
    # test_SviPowerLaw()
    # test_LatticePricer()
    # test_DeAmericanize()
    # test_AmericanVolSurface()
    test_AmericanVolSurface_data()
    # test_LatticePricerAccuracy_FlatVol()
    # test_LatticePricerAccuracy_SpxVol()
    # test_LatticePricerAccuracy_SpxVol_finetune()
    # test_LatticePricerAccuracy_SpxVol_deAm()
    # test_LatticePricer_ATMEuPut()
    # test_LatticePricer_ATMEuCall()
    # test_LatticePricer_ATMAmPut()
    # test_LatticePricer_ATMAmCall()
    # test_LatticePricer_Speedup()
