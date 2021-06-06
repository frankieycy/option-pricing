import os
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import BDay

def makeDirectory(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def getCurrentTime():
    return datetime.today().strftime("%Y%m%d %T")

def getPrevBDay(n=1):
    today = datetime.today()
    return (today-BDay(n)).strftime("%Y-%m-%d")

def bDaysBetween(d1, d2):
    return len(pd.bdate_range(d1,d2))
