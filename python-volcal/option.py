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

def GenerateImpVolDatasetFromStdDf(df):
    # Generate implied vol dataset from standardized options chain df
    # Fi,PVi are implied from put-call parity, for each maturity i
    # Columns: "Contract Name","Expiry","Texp","Put/Call","Strike","Bid","Ask"
    Texp = df['Texp'].unique()
    for T in Texp:
        dfT = df[df['Texp']==T]
        dfTc = dfT[dfT['Put/Call']=='Call']
        dfTp = dfT[dfT['Put/Call']=='Put']
        Kc = dfTc['Strike']
        Kp = dfTp['Strike']
        K = Kc[Kc.isin(Kp)]
        dfTc = dfTc[Kc.isin(K)]
        dfTp = dfTp[Kp.isin(K)]
        dfTc['Mid'] = (dfTc['Bid']+dfTc['Ask'])/2
        dfTp['Mid'] = (dfTp['Bid']+dfTp['Ask'])/2
        if len(K) >= 6:
            pass
