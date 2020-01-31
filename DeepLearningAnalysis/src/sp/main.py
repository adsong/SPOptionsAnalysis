# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:54:20 2020

@author: Lee Sak Park
"""

from ai_modeling import AI
import pandas as pd
import numpy as np
from option_analysis import PO
import datetime as dt

class Prediction():
    
    def __init__(self):
        pass
        
    def gen_prediction(self, rate = 20):
        if dt.datetime.today().weekday() >  4:
            number = 15
        else:
            number = 15 - dt.datetime.today().weekday() - 1

        x = AI()
        x.data_prep(number, rate)
        result = []
        for r in range(18, 28):
            x.run_ai(r)
            result.append(x.y_pred.round(2))
        result = np.array(result)
        self.count = len(result[result > .5]) / len(result)
        self.prob = result.mean()
    
    def gen_prediction_table(self, start = 0, end = 20):
        rg = range(end-start+1)
        df = pd.DataFrame({"rates": range(start, end+1)})
        df['probability'] = np.nan
        df['ratio'] = np.nan 
        for r in rg:
            self.gen_prediction( rate = r)
            df['probability'][r] = self.prob
            df['ratio'][r] = self.count
        self.prediction_table_dl = df
    
    
    def combine_prediction(self, rate = 20):
        self.gen_prediction( rate)
        ai_result = (self.prob, self.count)
        y = PO()
        dic = y.prediction
        dic['ai'] = ai_result
        self.prediction = dic
        return(self.prediction)
        
        
    def combine_prediction_table(self, s = 0, e = 20):
        if dt.datetime.today().weekday() >  4:
            number = 15
        else:
            number = 15 - dt.datetime.today().weekday() - 1
        self.gen_prediction_table(s, e)
        df = self.prediction_table_dl
        x = PO()
        x.prediction_with_variables(s, e, td = number)
        df2 = x.predictions
        
        self.combined_prediction_table = pd.concat([df, df2], axis = 1)
        return(self.combined_prediction_table)


if __name__ == "__main__":

    x = Prediction()
    x.combine_prediction()
    #print(x.combine_prediction())
