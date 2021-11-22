import os
import pandas as pd
from dateutil import parser
from datetime import datetime
from pandas.tseries.offsets import BDay

def isInDirectory(file, dir="./"):
    return os.path.isfile(dir+file)

def makeDirectory(dir):
    if not os.path.exists(dir): os.makedirs(dir)

def getCurrentTime():
    return datetime.today().strftime("%Y%m%d %T")

def getPrevBDay(n=1):
    today = datetime.today()
    return (today-BDay(n)).strftime("%Y-%m-%d")

def getRecentBDay():
    today = datetime.today()
    bDay = today-BDay(0)
    if bDay>today: bDay = today-BDay(1)
    return bDay.strftime("%Y-%m-%d")

def bDaysBetween(d1, d2):
    return len(pd.bdate_range(d1,d2))

def bDaysFromToday(date):
    today = datetime.today().strftime("%Y-%m-%d")
    return bDaysBetween(today,date)

def parseStringDateToFormat(date, fmt):
    return parser.parse(date).strftime(fmt)
