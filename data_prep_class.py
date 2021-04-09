# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:19:34 2021

@author: romainb
"""


# Load a local copy of the current ODYM branch:
import sys
import os
import pandas as pd
import pickle
import xlrd
from copy import deepcopy
import logging as log
        
    
# add ODYM module directory to system path, relative
MainPath = os.path.join('..', 'odym', 'modules')
sys.path.insert(0, MainPath)

# add ODYM module directory to system path, absolute
sys.path.insert(0, os.path.join(os.getcwd(), 'odym', 'modules'))

# Specify path to dynamic stock model and to datafile, relative
DataPath = os.path.join('..', 'data')

# Specify path to dynamic stock model and to datafile, absolute
DataPath = os.path.join(os.getcwd(), 'data')


import ODYM_Classes as msc # import the ODYM class file
import ODYM_Functions as msf # import the ODYM function file
import mfa_system # import the system definition
        
class DataPrep(object):
   
    


    def __init__(self, DataPath, Mylog, pickle_parameters=None):
        self.DataPath = DataPath
        self.Mylog = Mylog
        self.pickle_parameters = pickle_parameters
        self.initial_config()
        self.build_index_table()
        print('Read model data and parameters.')
        self.get_parameter_dict()


    def initial_config(self):
        self.Mylog.info('### 2 - Load Config file and read model control parameters')
        #Read main script parameters
        #Load project-specific config file
        self.ProjectSpecs_Name_ConFile = 'ODYM_Config_Al_cars.xlsx'
        self.Model_Configfile = xlrd.open_workbook(os.path.join(self.DataPath, self.ProjectSpecs_Name_ConFile))
        self.ScriptConfig = {'Model Setting': self.Model_Configfile.sheet_by_name('Config').cell_value(3,3)}
        self.Model_Configsheet = self.Model_Configfile.sheet_by_name('Setting_' + self.ScriptConfig['Model Setting'])
        
        self.Name_Scenario            = self.Model_Configsheet.cell_value(3,3)
        print(self.Name_Scenario)
        
        ### 1.2) Read model control parameters
        #Read control and selection parameters into dictionary
        self.ScriptConfig = msf.ParseModelControl(self.Model_Configsheet,self.ScriptConfig) 
        print(self.ScriptConfig)
        
        
        self.Mylog.info('### 3 - Read classification and data')
        # This is standard for each ODYM model run.
        # Read model run config data
        self.Classfile  = xlrd.open_workbook(os.path.join(self.DataPath, 
                                                     str(self.ScriptConfig['Version of master classification']) \
                                                     + '.xlsx'))
        self.Classsheet = self.Classfile.sheet_by_name('MAIN_Table')
        
        self. MasterClassification = msf.ParseClassificationFile_Main(self.Classsheet,self.Mylog)
        
        self.IT_Aspects,self.IT_Description,self.IT_Dimension,self.IT_Classification,self.IT_Selector,\
        self.IT_IndexLetter,self.PL_Names,self.PL_Description,self.PL_Version,self.PL_IndexStructure, \
        self.PL_IndexMatch,self.PL_IndexLayer,self.PrL_Number,self.PrL_Name,self.PrL_Comment,self.PrL_Type,ScriptConfig = \
            msf.ParseConfigFile(self.Model_Configsheet,self.ScriptConfig,self.Mylog)
            
            
    
            
        print('Define model classifications and select items for model classifications according to information provided by config file.')
        self.ModelClassification  = {} # Dict of model classifications
        for m in range(0,len(self.IT_Aspects)):
            self.ModelClassification[self.IT_Aspects[m]] = deepcopy(self.MasterClassification[self.IT_Classification[m]])
            EvalString = msf.EvalItemSelectString(self.IT_Selector[m],len(self.ModelClassification[self.IT_Aspects[m]].Items))
            if EvalString.find(':') > -1: # range of items is taken
                RangeStart = int(EvalString[0:EvalString.find(':')])
                RangeStop  = int(EvalString[EvalString.find(':')+1::])
                self.ModelClassification[self.IT_Aspects[m]].Items = self.ModelClassification[self.IT_Aspects[m]].Items[RangeStart:RangeStop]           
            elif EvalString.find('[') > -1: # selected items are taken
                self.ModelClassification[self.IT_Aspects[m]].Items = \
                    [self.ModelClassification[self.IT_Aspects[m]].Items[i] for i in eval(EvalString)]
            elif EvalString == 'all':
                None
            else:
                self.Mylog.info('ITEM SELECT ERROR for aspect ' + self.IT_Aspects[m] + ' were found in datafile.</br>')
                break
            
    def build_index_table(self):    
        # Define model index table and parameter dictionary
        self.Model_Time_Start = int(min(self.ModelClassification['Time'].Items))
        self.Model_Time_End   = int(max(self.ModelClassification['Time'].Items))
        self.Model_Duration   = self.Model_Time_End - self.Model_Time_Start
        
        print('Define index table dataframe.')
        self.IndexTable = pd.DataFrame({'Aspect'        : self.IT_Aspects, # 'Time' and 'Element' must be present!
                                   'Description'   : self.IT_Description,
                                   'Dimension'     : self.IT_Dimension,
                                   'Classification': [self.ModelClassification[Aspect] for Aspect in self.IT_Aspects],
                                   # Unique one letter (upper or lower case) indices to be used later for calculations.
                                   'IndexLetter'   : self.IT_IndexLetter}) 

        # Default indexing of IndexTable, other indices are produced on the fly
        self.IndexTable.set_index('Aspect', inplace = True) 
        
        # Add indexSize to IndexTable:
        self.IndexTable['IndexSize'] = \
            pd.Series([len(self.IndexTable.Classification[i].Items) for i in range(0,len(self.IndexTable.IndexLetter))], index=self.IndexTable.index)
        
        # list of the classifications used for each indexletter
        self.IndexTable_ClassificationNames = [self.IndexTable.Classification[i].Name for i in range(0,len(self.IndexTable.IndexLetter))] 

        #Define shortcuts for the most important index sizes:
        self.Nt = len(self.IndexTable.Classification[self.IndexTable.index.get_loc('Time')].Items)
        self.Nr = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('r')].Items)
        self.Np = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('p')].Items) 
        self.Ns = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('s')].Items) 
        self.Nz = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('z')].Items)
        self.Na = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('a')].Items)
        self.NS = len(self.IndexTable.Classification[self.IndexTable.set_index('IndexLetter').index.get_loc('S')].Items)



    def get_parameter_dict(self):
        if self.pickle_parameters:
            self.ParameterDict = pickle.load(open(self.pickle_parameters, "rb" ))
        else:            
            self.ParameterDict = {}
            for mo in range(0,len(self.PL_Names)):
                ParPath = os.path.join(self.DataPath,self.PL_Version[mo])
                print('Reading parameter ' + self.PL_Names[mo])
                # Do not change order of parameters handed over to function!
                MetaData, Values, Uncertainty = msf.ReadParameterV2(ParPath, self.PL_Names[mo], self.PL_IndexStructure[mo], 
                                                     self.PL_IndexMatch[mo], self.PL_IndexLayer[mo],
                                                     self.MasterClassification, self.IndexTable,
                                                     self.IndexTable_ClassificationNames, self.ScriptConfig, self.Mylog, ParseUncertainty = True) 
                self.ParameterDict[self.PL_Names[mo]] = msc.Parameter(Name = MetaData['Dataset_Name'], 
                                                            ID = MetaData['Dataset_ID'], 
                                                            UUID = MetaData['Dataset_UUID'],
                                                            P_Res = None,
                                                            MetaData = MetaData,
                                                            Indices = self.PL_IndexStructure[mo], 
                                                            Values=Values, 
                                                            Uncert=Uncertainty,
                                                                    Unit = MetaData['Dataset_Unit'])
            # Export ParameterDict to  pickle file for easier loading next time
            file_name = "ParameterDict.p"
            pickle.dump(self.ParameterDict, open(file_name, "wb" ))
            print("ParameterDict exported to: ", file_name)
    
    
    def create_mfa_system(self):
        self.PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
                          Geogr_Scope = 'World', 
                          Unit = 'Mt', 
                          ProcessList = [], 
                          FlowDict = mfa_system.FlowDict, 
                          StockDict = mfa_system.StockDict,
                          ParameterDict = self.ParameterDict, 
                          Time_Start = self.Model_Time_Start, 
                          Time_End = self.Model_Time_End, 
                          IndexTable = self.IndexTable, 
                          Elements = self.IndexTable.loc['Element'].Classification.Items, 
                          Graphical = None) # Initialize MFA system
        
        # Check Validity of index tables:
        # returns true if dimensions are OK and time index is present and element list is not empty
        self.PassengerVehicleFleet_MFA_System.IndexTableCheck() 
        
        # Add processes to system
        for m in range(0, len(self.PrL_Number)):
            self.PassengerVehicleFleet_MFA_System.ProcessList.append(msc.Process(Name = self.PrL_Name[m], ID   = self.PrL_Number[m]))
            
        self.PassengerVehicleFleet_MFA_System.Initialize_StockValues() # Assign empty arrays to stocks according to dimensions.
        self.PassengerVehicleFleet_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions. 
        return self.PassengerVehicleFleet_MFA_System
    
    
if __name__ == "__main__":
    # Initialize loggin routine
    log_verbosity = eval("log.DEBUG")
    log_filename = 'LogFileTest.md'
    log.getLogger('matplotlib').setLevel(log.WARNING)
    [Mylog, console_log, file_log] = msf.function_logger(log_filename, os.getcwd(),
                                                          log_verbosity, log_verbosity)
    Mylog.info('### 1. - Initialize.')
               
    Data_Prep = DataPrep(DataPath,Mylog)
    # pickle.dump(DataPrep, open( "DataPrep.p", "wb" ) )

