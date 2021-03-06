# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 02:36:03 2020

@author: Lee Sak Park
"""

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


class PO():


    def __init__(self):
        self.get_vix_spot()
        self.get_vix_3m()
        self.get_uvxy()
        self.do_it_all()
    
    def do_it_all(self, trading_days = 10, drops = 20):
        self.get_clean_data(td = trading_days, how_much =drops)
        self.transform()
        self.data_for_analysis()
        self.analysis()
        self.get_scores()
        self.final_prediction()
        
        
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
                if (dt.datetime.today().time().hour < 20) & (dt.datetime.today().time().hour > 9):
                    last_line_vix3m = web.DataReader("^VIX3M", 'yahoo')
                    last_line_vix3m.columns = last_line_vix3m.columns.str.lower().str.replace(" ","_")
                    last_line_vix3m = last_line_vix3m[['open', 'high','low', 'close']]
                    vix_3m = pd.concat([vix_3m, last_line_vix3m])
            
            # standardize where the open and closing prices stands between high and low price of the day.
            vix_3m['ohl'] = (vix_3m['open'] - vix_3m['low'])/(vix_3m['high']- vix_3m['low']) #ohl : open price between high and low
            vix_3m['chl'] = (vix_3m['close'] - vix_3m['low'])/(vix_3m['high']- vix_3m['low']) #chl: close price between high and low
            
            # ohl and chl in previous trading days
            # ohl1 : ohl of the last trading day
            
            for r, s, t in zip(['ohl1', 'ohl2','ohl3'],['chl1','chl2','chl3'], range(3)):
                vix_3m[r] = np.nan
                vix_3m[s] = np.nan
                for i in range(len(vix_3m)-(t+1)):
                    vix_3m[r][i+(t+1)] = vix_3m['ohl'][i]
                    vix_3m[s][i+(t+1)] = vix_3m['chl'][i]
        
            vix_3m = vix_3m.iloc[:,[1,4,5,6,7,8,9,10,11]]
            vix_3m.rename(columns = {'high':'vix_3m_high'}, inplace = True)
            
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
    
    
        # uvxy.diff_from_min is how different the closing price is from the minimum of last 8 trading days
        uvxy['diff_from_min'] = np.nan
    
    
        for r in range(len(uvxy)-8):
            uvxy['diff_from_min'][r+8] = round((uvxy['adj_close'][r+8]-uvxy['adj_close'][r:r+8].min())/uvxy['adj_close'][r:r+8].min()*100,2)
        
        # diff_from_max
            # diff_from_max1 = 3 days / 3
            # diff_from_max2 = 15 days / 15
        uvxy['diff_from_max1'] = np.nan
        uvxy['diff_from_max2'] = np.nan
    
        for r in range(len(uvxy)-3):
            uvxy['diff_from_max1'][r+3] = round((uvxy['adj_close'][r+3]-uvxy['adj_close'][r:r+3].max())/uvxy['adj_close'][r+3]*100/3,2)
        for r in range(len(uvxy)-15):
            uvxy['diff_from_max2'][r+15] = round((uvxy['adj_close'][r+15]-uvxy['adj_close'][r:r+15].max())/uvxy['adj_close'][r+15]*100/15,2)
        
        # we need to take logarithm to make it normally distributed
        uvxy['diff_from_min'] =  np.log(uvxy['diff_from_min'] - uvxy['diff_from_min'].min() + 1)
        self.uvxy = uvxy

    def get_clean_data(self, td = 10, how_much = 20):
        vix_spot = self.vix_spot.copy()
        uvxy = self.uvxy.copy()
        vix_3m = self.vix_3m.copy()
        # uvxy.dff_afterward is how different the closing price is from the minimum of next td trading days
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
        vix['spread'] = round((vix['spot_high']-vix['vix_3m_high'])/vix['vix_3m_high']*100, 2)
    
        # save the most recent before dropping
        self.last = vix.tail(1)
        
        # drop rows with np.nan
        vix.dropna(inplace = True)

        # defining a function
    
        def over(x, k = how_much):
            if x >k:
                return(1)
            else:
                return(0)
        # vix.opportunity is telling us it was an opportunity to buy the put option
        vix['opportunity'] = vix.diff_afterward.apply(over)
        self.vix = vix
        self.max_trading_days = td
        self.threshold = how_much


    def transform(self):
        vix = self.vix.copy()
        last = self.last.copy()
        vix['x1'] = vix['vix_3m_high']*vix['diff_from_max2']
        vix['x2'] =  np.log(vix['vix_3m_high']*vix['diff_from_min'])
        vix['x3'] = vix['vix_3m_high']*vix['x2']
        vix['x4'] = vix['diff_from_max2']*vix['diff_from_min']
        vix['x5'] = vix['diff_from_max2']*vix['x2']
        vix.drop(['spot_close','spot_high', 'chl','chl1','ohl1','ohl2', 'diff_from_max1',
                  'chl2','ohl3','chl3','close','c1','c2','c3'], axis = 1, inplace = True)
        self.vix = vix
        last['x1'] = last['vix_3m_high']*last['diff_from_max2']
        last['x2'] =  np.log(last['vix_3m_high']*last['diff_from_min'])
        last['x3'] = last['vix_3m_high']*last['x2']
        last['x4'] = last['diff_from_max2']*last['diff_from_min']
        last['x5'] = last['diff_from_max2']*last['x2']
        last = last[['vix_3m_high','diff_from_max2','diff_from_min', 
                   'spread', 'x1', 'x2', 'x3', 'x4', 'x5']]
        self.last = last
        
    def data_for_analysis(self):
        vix = self.vix.copy()
        
        # singling out the diff_afterward column from vix to analyze after prediction is made
        actual_diff = vix.loc[:,'diff_afterward']
        self.actual_diff = actual_diff.tail(100)
        
        
        # for the modeling, we don't use diff_afterward
        vix.drop(['diff_afterward'], axis = 1, inplace = True)
        vix = vix[['vix_3m_high','diff_from_max2','diff_from_min', 
                   'spread', 'x1', 'x2', 'x3', 'x4', 'x5', 'opportunity']]
        X = vix.iloc[:,0:-1]
        y = vix.iloc[:,-1]
        
        self.X = X
        self.y = y
        
        
        self.X_train = X.head(len(y) - 100)
        self.X_test = X.tail(100)
        self.y_train= y.head(len(y) - 100)
        self.y_test  =y.tail(100)
    
    
    
    def analysis(self):
        self.data_for_analysis()
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test 
        actual_diff = self.actual_diff
        
        X = self.X
        y = self.y
        
        
        ### Gradient Boosted Regression Trees
       
        gbrt = GradientBoostingClassifier()
        gbrt.fit(X_train, y_train)
        gbrt_pred = gbrt.predict(X_test)
        gbrt_accuracy = gbrt.score(X_test, y_test)
        self.gbrt_pred_raw = gbrt_pred
        self.gbrt_pred = actual_diff[np.argwhere(gbrt_pred==1)]
        gbrt = GradientBoostingClassifier()
        self.gbrt_model = gbrt.fit(X, y)
       
        
        
        ### Random Forest Classifier
        forest = RandomForestClassifier(n_estimators = 25)
        forest.fit(X_train, y_train)
        forest_pred = forest.predict(X_test)
        forest_accuracy = forest.score(X_test, y_test)
        self.forest_pred_raw = forest_pred
        self.forest_pred = actual_diff[np.argwhere(forest_pred==1)]
        forest = RandomForestClassifier(n_estimators = 25)
        self.forest_model = forest.fit(X,y)
    
        ### SVC
        svc = SVC(gamma = 'auto')
        svc.fit(X_train, y_train)
        svc_pred = svc.predict(X_test)
        svc_accuracy = svc.score(X_test, y_test)
        self.svc_pred_raw = svc_pred
        self.svc_pred = actual_diff[np.argwhere(svc_pred==1)]
        svc = SVC(gamma = 'auto')
        self.svc_model = svc.fit(X, y)
        
        #### self.accuracy (it is not the score)
        tests = ["gbrt",'forest','svc']
        self.accuracy = dict(zip(tests, [gbrt_accuracy, forest_accuracy, svc_accuracy]))
        
        
        # combined
        combined = []
        for r in range(len(svc_pred)):
            if (forest_pred[r] ==1)& (svc_pred[r] ==1) & (gbrt_pred[r]==1):
                combined.append(1)
            else:
                combined.append(0)
        self.combined_raw = np.array(combined)
        self.combined_pred = actual_diff[np.argwhere(np.array(combined)==1)]
        

    
    def get_scores(self):
        ### calculating scores
        def Calc_score(vec1):
            vec2 = self.y_test
            result = []
            for r in range(len(vec2)):
                if(vec1[r]==1)& (vec2[r] == 1):
                    result.append(1)
                elif(vec1[r]==1)&(vec2[r]==0):
                    result.append(0)
            num = sum(result)
            den = len(result)
            if den == 0:
                return(np.nan)
            else:
                return(num/den)
        
        tests = ["gbrt",'forest','svc', 'combined']
        scores = [Calc_score(self.gbrt_pred_raw), Calc_score(self.forest_pred_raw),
                   Calc_score(self.svc_pred_raw), Calc_score(self.combined_raw)]  
        self.results = dict(zip(tests, scores))
        
    def final_prediction(self):
        tests = ["gbrt",'forest','svc']
        final = [self.gbrt_model.predict(self.last)[0], self.gbrt_model.predict(self.last)[0], self.svc_model.predict(self.last)[0]]
        self.prediction = dict(zip(tests, final))
        


        
    def print_results(self):
        means = [self.gbrt_pred.mean(), self.forest_pred.mean(), self.svc_pred.mean(), self.combined_pred.mean()]
        for r in range(len(self.results)):
            print("score of {0} is : {1:.2f} \n  -> mean price drop is : {2: .2f}".format(
                list(self.results.keys())[r], list(self.results.values())[r], means[r]))
        for r in range(len(self.prediction)):
            print("the prediction of {0} is : {1}".format(
                list(self.prediction.keys())[r], list(self.prediction.values())[r]))

    def accuracy_analysis(self, bootstrap = 500):
        accuracies = np.repeat(np.nan, 3)
        scores = np.repeat(np.nan, 4)
        
        for r in range(bootstrap):
            self.analysis()
            self.get_scores()
            accuracies = np.vstack((accuracies, np.array(list(self.accuracy.values()))))
            scores = np.vstack((scores, np.array(list(self.results.values()))))
        
        #dropping rows with np.nan
        mask_accuracies= np.all(np.isnan(accuracies), axis=1)
        mask_scores = np.all(np.isnan(scores), axis = 1)
        
        
        self.accuracies = accuracies[~mask_accuracies]
        self.scores = np.delete(scores[~mask_scores], 3,1)

    def plotting_accuracies(self):
        fig = plt.figure(figsize=(20,5))
        df = pd.DataFrame(data = self.accuracies, columns = ["gbrt",'forest','svc'])
        fig, ax = plt.subplots()
        ax.boxplot(df.values)
        plt.show()
    
    def plotting_scores(self):
        fig = plt.figure(figsize=(20,5))
        df = pd.DataFrame(data = self.scores, columns = ["gbrt",'forest','svc']).dropna(axis = 0)
        fig, ax = plt.subplots()
        ax.boxplot(df.values)
        plt.show()
    
    
    # It generates dataframe showing the predictions for the range of drop argument
    def prediction_with_variables(self, begin, end, td = 10):
        a = begin
        b = end+1
        results = np.repeat(np.nan, 3)
        scores = np.repeat(np.nan, 4)
        for r in range(a, b):
            self.do_it_all(drops = r, trading_days = td)
            scores = np.vstack((scores, np.array(list(self.results.values()))))
            results= np.vstack((results, np.array(list(self.prediction.values()))))
        results = results[1:]
        scores = scores[1:]
        results = pd.DataFrame(data = results, index = range(a, b),
                               columns =  ["gbrt",'forest','svc'])
        scores = pd.DataFrame(data = scores, index = range(a, b),
                               columns =  ["gbrt_score",'forest_score',
                                           'svc_score', 'combined_score'])
        outcome = pd.merge(left = results, right= scores, 
                           left_index = True, right_index = True)
        self.predictions = outcome
        
    
    # last_quarter runs a simulation over the last quarters and show what was 
    # the actual results when each model gave the signal
    def last_quarter(self, trading_days = 10, rate  = 20, fail = 7):
        self.get_clean_data(td = trading_days, how_much =rate)
        self.transform()
        self.data_for_analysis()
        self.analysis()
        self.get_scores()
        self.final_prediction()

        v1 = self.actual_diff
        mask_1 = self.gbrt_pred_raw == 1
        mask_2 = self.forest_pred_raw == 1
        mask_3 = self.svc_pred_raw == 1
        self.last_quarter_gbrt = v1[mask_1]
        self.last_quarter_forest = v1[mask_2]
        self.last_quarter_svc  = v1[mask_3]
        
        y1 = v1[mask_1]
        y2 = v1[mask_2]
        y3  = v1[mask_3]
        
        f = fail
        
        self.failure_gbrt = len(y1[y1<f])/len(y1)
        self.failure_forest = len(y2[y2<f])/len(y2)
        self.failure_svc = len(y3[y3<f])/len(y3)
        
        self.failure_gbrt_specific = y1[y1<f]
        self.failure_forest_specific = y2[y2<f]
        self.failure_svc_specific = y3[y3<f]
        
        
        
        
