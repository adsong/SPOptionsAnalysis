import urllib.request
import pandas as pd
from scipy import stats
import datetime as dt
import pandas_datareader.data as web
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class CreateVix():

    def __init__(self):
        self.get_vix_spot()
        self.get_vix_3m()
        self.get_uvxy()

    def get_vix_spot(self):
        # we import the other the data from yahoo finance
        vix_spot = web.DataReader("^VIX", 'yahoo', start = dt.datetime(2012,12,31))

        # formatting column names
        vix_spot.columns = vix_spot.columns.str.lower().str.replace(" ","_")

        # dropping unnecessaries and changing colnames
        vix_spot = vix_spot[['close','high']]
        vix_spot.rename(columns = {'close':'spot_close','high':'spot_high'}, inplace = True)

        self.vix_spot = vix_spot

    def get_vix_3m(self):

        # ^vix3m is imported from csv data from cboe website
        # However, the data for today is not updated frequently...

        vix_3m = pd.read_csv(urllib.request.urlopen(
            'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vix3mdailyprices.csv'),
            skiprows = 2, index_col = 0)
        vix_3m = vix_3m.loc['12/31/2012':,:]

        # change the type of index to datetime
        vix_3m.index = pd.to_datetime(pd.Series(vix_3m.index), format = "%m/%d/%Y")

        # formatting column names
        vix_3m.columns = vix_3m.columns.str.lower().str.replace(" ","_")

        # if it is during the trading hours, we get the last row of vix_3m from yahoo finance
        if dt.datetime.today().weekday() in range(5):
            if vix_3m.tail(1).index[0] ==(dt.datetime.now()- dt.timedelta(days = 1)).strftime("%m/%d/%Y"):
                last_line_vix3m = web.DataReader("^VIX3M", 'yahoo')
                last_line_vix3m.columns = last_line_vix3m.columns.str.lower().str.replace(" ","_")
                last_line_vix3m = last_line_vix3m[['open', 'high','low', 'close']]
                vix_3m = pd.concat([vix_3m, last_line_vix3m])

        # we are going to create variables that will compare opening price to highest price which will be denoted as "oh"
        # we are going to create variables that will compare closing price to highest price which will be denoted as "ch"
        # we are going to create variables that will compare highest price to lowest price which will be denoted as "lh"


        vix_3m['oh'] = (vix_3m['high'] - vix_3m['open'])/(vix_3m['open'])
        vix_3m['ch'] = (vix_3m['high'] - vix_3m['close'])/(vix_3m['close'])
        vix_3m['lh']= (vix_3m['high'] - vix_3m['low'])/(vix_3m['low'])

        # ohl and chl in previous trading days
        # ohl1 : ohl of the last trading day
        for a,b,c,m, n in zip(np.repeat("oh_", 10),np.repeat("ch_", 10),np.repeat("lh_", 10),list('0123456789'), range(9)):
            name1 = a+m
            name2 = b+m
            name3 = c+m


            vix_3m[name1] = np.nan
            vix_3m[name2]= np.nan
            vix_3m[name3] = np.nan
            vix_3m[name1] = vix_3m['oh'] - vix_3m['oh'].rolling(window = n+2, min_periods=n+2).mean()
            vix_3m[name2]= vix_3m['ch']- vix_3m['ch'].rolling(window = n+2, min_periods=n+2).mean()
            vix_3m[name3]= vix_3m['lh']- vix_3m['ch'].rolling(window = n+2, min_periods=n+2).mean()

        self.vix_3m = vix_3m



    def get_uvxy(self):
        uvxy = web.DataReader("UVXY", 'yahoo', start = dt.datetime(2012,12,31))
        uvxy.columns = uvxy.columns.str.lower().str.replace(" ","_")

        # the structure changed as of 2/27/2018


        # uvxy.change is the price change in percentage
        # c1 = change from last day
        # c2 = change from two days ago
        # c3 = change from three days ago
        cs = ['c1','c2','c3']

        for r in range(len(cs)):
            uvxy[cs[r]] = np.nan
            for i in range(len(uvxy)-(r+1)):
                uvxy[cs[r]][i+(r+1)] = round((uvxy['adj_close'][i+(r+1)] - uvxy['adj_close'][i])/uvxy['adj_close'][i]*100 ,2)



        for d, m, n in zip(np.repeat("diff_from_min_", 8), list('23456789'), range(7) ):
            name = d+m
            uvxy[name] = (uvxy['adj_close'] - uvxy['adj_close'].rolling(window = n+2, min_periods = n+2).min())/uvxy['adj_close'].rolling(window = n+2, min_periods = n+2).min()


        # diff_from_max
            # diff_from_max1 = 5 days / 5
            # diff_from_max2 = 15 days / 15

        uvxy['diff_from_max1'] = np.nan
        uvxy['diff_from_max2'] = np.nan

        uvxy['diff_from_max1'] = uvxy['high'].rolling(window = 5, min_periods = 5).max()/uvxy['adj_close']
        uvxy['diff_from_max2'] = uvxy['high'].rolling(window = 15, min_periods = 15).max()/uvxy['adj_close']


        # we need to take logarithm to make it normally distributed
        uvxy['diff_from_min'] =  (uvxy['adj_close'] - uvxy['low'].rolling(window = 5, min_periods = 5).min())/uvxy['low'].rolling(window = 5, min_periods = 5).min()
        uvxy['diff_from_min'] = np.log(uvxy['diff_from_min'] - uvxy['diff_from_min'].min() + 1)
        self.uvxy = uvxy
