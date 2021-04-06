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
    
import ODYM_Classes as msc # import the ODYM class file
import ODYM_Functions as msf # import the ODYM function file
import mfa_system # import the system definition




class Data_Prep(object):

    


    def __init__(self, DataPath, Mylog):
        self.DataPath = DataPath
        self.Mylog = Mylog
        self.initial_config()


    def initial_config(self):
                
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
        self.PL_IndexMatch,self.PL_IndexLayer,self.PrL_Number,self.PrL_Name,self.PrL_Comment,self.PrL_Type,self.ScriptConfig = \
            msf.ParseConfigFile(self.Model_Configsheet,self.ScriptConfig,self.Mylog)
            
            
    
            
        print('Define model classifications and select items for model classifications according to information provided by config file.')
        self.ModelClassification  = {} # Dict of model classifications
        for m in range(0,len(self.IT_Aspects)):
            ModelClassification[IT_Aspects[m]] = deepcopy(MasterClassification[IT_Classification[m]])
            EvalString = msf.EvalItemSelectString(IT_Selector[m],len(ModelClassification[IT_Aspects[m]].Items))
            if EvalString.find(':') > -1: # range of items is taken
                RangeStart = int(EvalString[0:EvalString.find(':')])
                RangeStop  = int(EvalString[EvalString.find(':')+1::])
                ModelClassification[IT_Aspects[m]].Items = ModelClassification[IT_Aspects[m]].Items[RangeStart:RangeStop]           
            elif EvalString.find('[') > -1: # selected items are taken
                ModelClassification[IT_Aspects[m]].Items = \
                    [ModelClassification[IT_Aspects[m]].Items[i] for i in eval(EvalString)]
            elif EvalString == 'all':
                None
            else:
                Mylog.info('ITEM SELECT ERROR for aspect ' + IT_Aspects[m] + ' were found in datafile.</br>')
                break
            
    
# Define model index table and parameter dictionary
Model_Time_Start = int(min(ModelClassification['Time'].Items))
Model_Time_End   = int(max(ModelClassification['Time'].Items))
Model_Duration   = Model_Time_End - Model_Time_Start

print('Define index table dataframe.')
IndexTable = pd.DataFrame({'Aspect'        : IT_Aspects, # 'Time' and 'Element' must be present!
                           'Description'   : IT_Description,
                           'Dimension'     : IT_Dimension,
                           'Classification': [ModelClassification[Aspect] for Aspect in IT_Aspects],
                           # Unique one letter (upper or lower case) indices to be used later for calculations.
                           'IndexLetter'   : IT_IndexLetter}) 

# Default indexing of IndexTable, other indices are produced on the fly
IndexTable.set_index('Aspect', inplace = True) 

# Add indexSize to IndexTable:
IndexTable['IndexSize'] = \
    pd.Series([len(IndexTable.Classification[i].Items) for i in range(0,len(IndexTable.IndexLetter))], index=IndexTable.index)

# list of the classifications used for each indexletter
IndexTable_ClassificationNames = [IndexTable.Classification[i].Name for i in range(0,len(IndexTable.IndexLetter))] 

#Define shortcuts for the most important index sizes:
Nt = len(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)
Nr = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('r')].Items)
Np = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('p')].Items) 
Ns = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('s')].Items) 
Nz = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('z')].Items)
Na = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('a')].Items)
NS = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('S')].Items)



def get_parameter_dict(DataPath, PL_Names):
    ParameterDict = {}
    for mo in range(0,len(PL_Names)):
        ParPath = os.path.join(DataPath,PL_Version[mo])
        print('Reading parameter ' + PL_Names[mo])
        # Do not change order of parameters handed over to function!
        MetaData, Values, Uncertainty = msf.ReadParameterV2(ParPath, PL_Names[mo], PL_IndexStructure[mo], 
                                             PL_IndexMatch[mo], PL_IndexLayer[mo],
                                             MasterClassification, IndexTable,
                                             IndexTable_ClassificationNames, ScriptConfig, Mylog, ParseUncertainty = True) 
        ParameterDict[PL_Names[mo]] = msc.Parameter(Name = MetaData['Dataset_Name'], 
                                                    ID = MetaData['Dataset_ID'], 
                                                    UUID = MetaData['Dataset_UUID'],
                                                    P_Res = None,
                                                    MetaData = MetaData,
                                                    Indices = PL_IndexStructure[mo], 
                                                    Values=Values, 
                                                    Uncert=Uncertainty,
                                                    Unit = MetaData['Dataset_Unit'])
    return ParameterDict

print('Read model data and parameters.')
ParameterDict = get_parameter_dict(DataPath, PL_Names)



# print('Read process list from model config sheet.')
# PrLix = 0
# while True: # search for process list entry
#     if Model_Configsheet.cell_value(PrLix,1) == 'Process Group List':
#         break
#     else:
#         PrLix += 1
        
# PrL_Number         = []
# PrL_Name           = []
# PrL_Code           = []
# PrL_Type           = []
# PrLix += 2 # start on first data row
# while True:
#     if Model_Configsheet.cell_value(PrLix,2) != '':
#         try:
#             PrL_Number.append(int(Model_Configsheet.cell_value(PrLix,2)))
#         except:
#             PrL_Number.append(Model_Configsheet.cell_value(PrLix,2))
#         PrL_Name.append(Model_Configsheet.cell_value(PrLix,3))
#         PrL_Code.append(Model_Configsheet.cell_value(PrLix,4))
#         PrL_Type.append(Model_Configsheet.cell_value(PrLix,5))
#         PrLix += 1
#     else:
#         break  




# PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
#                       Geogr_Scope = 'World', 
#                       Unit = 'Mt', 
#                       ProcessList = [], 
#                       # FlowDict = mfa_system.FlowDict, 
#                       # StockDict = mfa_system.StockDict,
#                       # ParameterDict = ParameterDict, 
#                       Time_Start = Model_Time_Start, 
#                       Time_End = Model_Time_End, 
#                       IndexTable = IndexTable, 
#                       Elements = IndexTable.loc['Element'].Classification.Items, 
#                       Graphical = None) # Initialize MFA system
                      
# # Check Validity of index tables:
# # returns true if dimensions are OK and time index is present and element list is not empty
# PassengerVehicleFleet_MFA_System.IndexTableCheck() 

# # Add processes to system
# for m in range(0, len(PrL_Number)):
#     PassengerVehicleFleet_MFA_System.ProcessList.append(msc.Process(Name = PrL_Name[m], ID   = PrL_Number[m]))
    
# PassengerVehicleFleet_MFA_System.Initialize_StockValues() # Assign empty arrays to stocks according to dimensions.
# PassengerVehicleFleet_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions. 


def create_mfa_system(IndexTable, FlowDict, StockDict, ParameterDict):
    PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
                      Geogr_Scope = 'World', 
                      Unit = 'Mt', 
                      ProcessList = [], 
                      FlowDict = FlowDict, 
                      StockDict = StockDict,
                      ParameterDict = ParameterDict, 
                      Time_Start = Model_Time_Start, 
                      Time_End = Model_Time_End, 
                      IndexTable = IndexTable, 
                      Elements = IndexTable.loc['Element'].Classification.Items, 
                      Graphical = None) # Initialize MFA system
    
    # Check Validity of index tables:
    # returns true if dimensions are OK and time index is present and element list is not empty
    PassengerVehicleFleet_MFA_System.IndexTableCheck() 
    
    # Add processes to system
    for m in range(0, len(PrL_Number)):
        PassengerVehicleFleet_MFA_System.ProcessList.append(msc.Process(Name = PrL_Name[m], ID   = PrL_Number[m]))
        
    PassengerVehicleFleet_MFA_System.Initialize_StockValues() # Assign empty arrays to stocks according to dimensions.
    PassengerVehicleFleet_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions. 

    return PassengerVehicleFleet_MFA_System