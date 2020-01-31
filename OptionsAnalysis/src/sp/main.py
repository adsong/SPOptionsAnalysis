'''
Created on Jan 29, 2020

@author: Andrew
'''

from optionsAnalysis import PO
from executeOptionsAnalysis import OptionsExecutor

if __name__=="__main__":
    print("Start.")

    try:
        executor = OptionsExecutor()
        machineLearningModel = PO()
        
        if(executor.hasPosition):
            machineLearningModel.do_it_all(executor.daysUntilContractEnd, executor.targetDrop)
        else:
            machineLearningModel.do_it_all(executor.daysUntilFridayTwoWeeks, executor.targetDrop)
    except Exception as e:
        print("Exception thrown during instantiation of objects - {}".format(e))
    else:
        if(executor.isFirstRun):
            executor.cleanReport()
        
        returnMessage = executor.execute(machineLearningModel)
        executor.writeReport(returnMessage, machineLearningModel)
    
    print("Done.")
