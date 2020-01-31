from create_vix import CreateVix
import pandas as pd
import numpy as np


class Variable_selection():
    def __init__(self):
        self.import_data()
    
    
    def gen_data(self, trading_days = 10, drop = 20):
        self.merge_vix(td = trading_days)
        self.add_opportunity(rate = drop)
        self.transformation()
        self.remove_variable()

    def import_data(self):
        self.create_vix = CreateVix()

    def merge_vix(self, td):
        df = self.create_vix
        vix_spot = df.vix_spot.copy()
        uvxy = df.uvxy.copy()
        vix_3m = df.vix_3m.copy()

        uvxy['diff_afterward'] = np.nan
        for r in range(len(uvxy)-td):
            uvxy.loc[:,'diff_afterward'].iloc[r] = round((1-uvxy.loc[:,'adj_close'].iloc[r+1:(r + td)].min()/uvxy.loc[:,'adj_close'].iloc[r])*100,2)

        uvxy = uvxy[['adj_close','c1','c2','c3','diff_from_max1','diff_from_max2','diff_from_min','diff_afterward']]
        uvxy.rename(columns = {'adj_close':'close'}, inplace = True)

        # term structure changed as of 2/27/2018
        uvxy.loc[:'2018-02-27',
                 ['close', 'c1','c2','c3', 'diff_from_max1','diff_from_max2',
                  'diff_afterward','diff_from_min']
                 ] = uvxy.loc[:'2018-02-27',
                              ['close', 'c1','c2','c3',
                               'diff_afterward','diff_from_min']]*.75




        # joining data
        vix = vix_spot.join(vix_3m,
                            lsuffix = "", rsuffix = "_3m"
                        ).join(uvxy, lsuffix = "_spot", rsuffix = "_uvxy")

        #vix.spread is the spread between vix.high_spot and vix.high_3m
        vix['spread'] = round((vix['spot_high']-vix['high'])/vix['high']*100, 2)

        # save the most recent before dropping
        self.today = vix.tail(1)
        today_ANN = vix.tail(1)
        today_ANN.drop(['close_uvxy','diff_afterward'], axis = 1, inplace = True)
        self.ANN_today = today_ANN

        # drop rows with np.nan
        vix.dropna(inplace = True)


        self.vix = vix

    def add_opportunity(self, rate):
        vix = self.vix.copy()
        actual_diff = vix.loc[:,'diff_afterward']
        self.actual_diff = actual_diff.tail(100)

        def over(x, k = rate):
            if x >k:
                return(1)
            else:
                return(0)
        # vix.opportunity is telling us it was an opportunity to buy the put option
        vix['opportunity'] = vix.diff_afterward.apply(over)

        vix.drop(['close_uvxy'], axis = 1, inplace  = True)
        self.vix = vix
        
        # tables for ANN
        df_ANN = vix[['spot_close', 'spot_high', 'open', 'high', 'low', 'close_spot', 'oh',
       'ch', 'lh', 'oh_0', 'ch_0', 'lh_0', 'oh_1', 'ch_1', 'lh_1', 'oh_2',
       'ch_2', 'lh_2', 'oh_3', 'ch_3', 'lh_3', 'oh_4', 'ch_4', 'lh_4', 'oh_5',
       'ch_5', 'lh_5', 'oh_6', 'ch_6', 'lh_6', 'oh_7', 'ch_7', 'lh_7', 'oh_8',
       'ch_8', 'lh_8', 'c1', 'c2', 'c3', 'diff_from_max1', 'diff_from_max2',
       'diff_from_min', 'spread', 'opportunity']]
        
        X = df_ANN.iloc[:,0:-1]
        y = df_ANN.iloc[:,-1]
        
        self.ANN_X = X
        self.ANN_y = y
        
        
        
        self.ANN_X_train = X.head(len(y) - 100)
        self.ANN_X_test = X.tail(100)
        self.ANN_y_train= y.head(len(y) - 100)
        self.ANN_y_test  =y.tail(100)

        
        
        

    def write_vix(self):
        self.vix.to_csv("vix_data.csv")

    def transformation(self):
        vix = self.vix.copy()
        last = self.today.copy()
        df = pd.concat([vix, last], sort = True)
        df['x1'] = df.low*df.oh_5
        df['x2'] = df.lh*df.oh_5
        df['x3'] = df.lh*df.diff_from_max1
        df['x4'] = df.ch_4*df.ch_4
        df['x5'] = df.oh_5*df.oh_5
        df['x6'] = df.oh_5*df.oh_8
        df['x7'] = df.oh_5*df.diff_from_max1
        df['x8'] =  df.oh_7*df.oh_7
        df['x9'] = df.oh_7*df.diff_from_max1
        df['x10'] = df.oh_8*df.oh_8
        df['x11'] = df.diff_from_max1*df.diff_from_max1
        self.concat = df
        self.vix = df.head(len(df)-1)
        self.today = df.tail(1)

    def remove_variable(self):
        vix = self.vix.copy()
        last = self.today.copy()
        


        vix = vix[['high', 'low','oh_5','oh_7','c1','diff_from_max1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11', 'opportunity', 'diff_afterward']]
        last = last[['high', 'low','oh_5','oh_7','c1','diff_from_max1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11']]
        self.vix = vix
        vix.drop(['diff_afterward'], axis = 1, inplace = True)
        X = vix.iloc[:,0:-1]
        y = vix.iloc[:,-1]
        
        self.X = X
        self.y = y
        
        
        self.X_train = X.head(len(y) - 100)
        self.X_test = X.tail(100)
        self.y_train= y.head(len(y) - 100)
        self.y_test  =y.tail(100)
        self.today = last
