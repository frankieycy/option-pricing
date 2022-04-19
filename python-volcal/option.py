import re
import datetime
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm
from time import time
from pricer import *

#### Options Chain #############################################################

def GenerateYfinOptionsChainDataset(fileName, underlying="^SPX"):
    # Generate options chain dataset from yahoo_fin
    # Bad data (unavailable bids/asks) for large maturities
    from yahoo_fin import options
    optionDates = options.get_expiration_dates(underlying)
    optionChains = []
    for date in tqdm(optionDates):
        try:
            chainPC = options.get_options_chain(underlying,date)
            # print(chainPC)
            for putCall in ["puts","calls"]:
                chain = chainPC[putCall]
                chain["Maturity"] = date
                chain["Put/Call"] = putCall
                chain = chain[["Maturity","Put/Call","Contract Name","Last Trade Date","Strike","Last Price","Bid","Ask","Change","% Change","Volume","Open Interest","Implied Volatility"]]
                optionChains.append(chain)
        except Exception: pass
    optionChains = pd.concat(optionChains)
    optionChains.to_csv(fileName, index=False)
    return optionChains

def StandardizeOptionsChainDataset(df, onDate):
    # Standardize options chain dataset
    onDate = parser.parse(onDate)
    cols = ["Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"]
    getDateFromContractName = lambda n: re.match(r'([a-z.]+)(\d+)([a-z])(\d+)',n,re.I).groups()[1]
    df["Expiry"] = pd.to_datetime(df["Contract Name"].apply(getDateFromContractName),format='%y%m%d')
    df["Texp"] = (df["Expiry"]-onDate).dt.days/365.25
    df = df[df['Texp']>0][cols].reset_index(drop=True)
    return df

#### Implied Vol Dataset #######################################################

def GenerateImpVolDatasetFromStdDf(df, Nntm=6, volCorrection=None):
    # Generate implied vol dataset from standardized options chain df
    # Fi,PVi are implied from put-call parity, for each maturity i
    # Columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"
    # Output: "Expiry","Texp","Strike","Bid","Ask","Fwd","CallMid","PV"
    Texp = df['Texp'].unique()
    ivdf = list()
    for T in Texp:
        dfT = df[df['Texp']==T].dropna()
        dfTc = dfT[dfT['Put/Call']=='Call']
        dfTp = dfT[dfT['Put/Call']=='Put']
        expiry = dfT['Expiry'].iloc[0]
        Kc = dfTc['Strike']
        Kp = dfTp['Strike']
        K = Kc[Kc.isin(Kp)] # common strikes
        dfTc = dfTc[Kc.isin(K)]
        dfTp = dfTp[Kp.isin(K)]
        dfTc['Mid'] = (dfTc['Bid']+dfTc['Ask'])/2 # add Mid col
        dfTp['Mid'] = (dfTp['Bid']+dfTp['Ask'])/2
        if len(K) >= Nntm:
            K = K.to_numpy()
            bidc = dfTc['Bid'].to_numpy()
            bidp = dfTp['Bid'].to_numpy()
            askc = dfTc['Ask'].to_numpy()
            askp = dfTp['Ask'].to_numpy()
            midc = dfTc['Mid'].to_numpy()
            midp = dfTp['Mid'].to_numpy()
            mids = midc-midp # put-call spread
            Kntm = K[np.argsort(np.abs(midc-midp))[:Nntm]] # ntm strikes
            i = np.isin(K,Kntm)
            def objective(params):
                F,PV = params
                return sum((mids[i]-PV*(F-K[i]))**2)
            opt = minimize(objective,x0=(np.mean(mids[i]+K[i]),1))
            F,PV = opt.x
            print(f"T={expiry.date()} F={F} PV={PV}")
            ivcb = BlackScholesImpliedVol(F,K,T,0,bidc/PV,"call")
            ivca = BlackScholesImpliedVol(F,K,T,0,askc/PV,"call")
            ivpb = BlackScholesImpliedVol(F,K,T,0,bidp/PV,"put")
            ivpa = BlackScholesImpliedVol(F,K,T,0,askp/PV,"put")
            if volCorrection == "delta": # wrong-spot correction
                dcb = BlackScholesDelta(F,K,T,0,ivcb,"call")
                dca = BlackScholesDelta(F,K,T,0,ivca,"call")
                dpb = BlackScholesDelta(F,K,T,0,ivpb,"put")
                dpa = BlackScholesDelta(F,K,T,0,ivpa,"put")
                ivb = (dcb*ivpb-dpb*ivcb)/(dcb-dpb)
                iva = (dca*ivpa-dpa*ivca)/(dca-dpa)
            else: # otm imp vols
                ivb = ivpb*(K<=F)+ivcb*(K>F)
                iva = ivpa*(K<=F)+ivca*(K>F)
            ivb[ivb<1e-8] = np.nan
            iva[iva<1e-8] = np.nan
            callb = BlackScholesFormula(F,K,T,0,ivb,"call")
            calla = BlackScholesFormula(F,K,T,0,iva,"call")
            callm = (callb+calla)/2
            ivdfT = pd.DataFrame({
                "Expiry":   expiry,
                "Texp":     T,
                "Strike":   K,
                "Bid":      ivb,
                "Ask":      iva,
                "Fwd":      F,
                "CallMid":  callm,
                "PV":       PV,
            })
            ivdf.append(ivdfT)
    ivdf = pd.concat(ivdf)
    return ivdf
