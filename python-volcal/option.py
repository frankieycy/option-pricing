import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

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
