# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:19:34 2021

@author: romainb
"""


# Load a local copy of the current ODYM branch:
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xlrd
import pylab
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
import dynamic_stock_model as dsm # import the dynamic stock model library
import custom_functions as cf
import mfa_system # import the system definition

# Initialize loggin routine



log_verbosity = eval("log.DEBUG")
log_filename = 'LogFileTest.md'
log.getLogger('matplotlib').setLevel(log.WARNING)
[Mylog, console_log, file_log] = msf.function_logger(log_filename, os.getcwd(),
                                                     log_verbosity, log_verbosity)
Mylog.info('### 1. - Initialize.')
           

Mylog.info('### 2 - Load Config file and read model control parameters')
#Read main script parameters
#Load project-specific config file
ProjectSpecs_Name_ConFile = 'ODYM_Config_Al_cars.xlsx'
Model_Configfile = xlrd.open_workbook(os.path.join(DataPath, ProjectSpecs_Name_ConFile))
ScriptConfig = {'Model Setting': Model_Configfile.sheet_by_name('Config').cell_value(3,3)}
Model_Configsheet = Model_Configfile.sheet_by_name('Setting_' + ScriptConfig['Model Setting'])

Name_Scenario            = Model_Configsheet.cell_value(3,3)
print(Name_Scenario)

### 1.2) Read model control parameters
#Read control and selection parameters into dictionary
ScriptConfig = msf.ParseModelControl(Model_Configsheet,ScriptConfig) 
print(ScriptConfig)






Mylog.info('### 3 - Read classification and data')
# This is standard for each ODYM model run.

# Read model run config data
Classfile  = xlrd.open_workbook(os.path.join(DataPath, 
                                             str(ScriptConfig['Version of master classification']) \
                                             + '.xlsx'))
Classsheet = Classfile.sheet_by_name('MAIN_Table')
ci = 1 # column index to start with
MasterClassification = {} # Dict of master classifications
while True:
    TheseItems = []
    ri = 10 # row index to start with    
    try: 
        ThisName = Classsheet.cell_value(0,ci)
        ThisDim  = Classsheet.cell_value(1,ci)
        ThisID   = Classsheet.cell_value(3,ci)
        ThisUUID = Classsheet.cell_value(4,ci)
        TheseItems.append(Classsheet.cell_value(ri,ci)) # read the first classification item
    except:
        print('End of file or formatting error while reading the classification file in column '+ str(ci) +'.')
        break
    while True:
        ri +=1
        try:
            ThisItem = Classsheet.cell_value(ri,ci)
        except:
            break
        if ThisItem != '':
            TheseItems.append(ThisItem)
    MasterClassification[ThisName] = msc.Classification(Name = ThisName, Dimension = ThisDim, 
                                                        ID = ThisID, UUID = ThisUUID, Items = TheseItems)
    ci +=1 
    
print('Read index table from model config sheet.')
ITix = 0
while True: # search for index table entry
    if Model_Configsheet.cell_value(ITix,1) == 'Index Table':
        break
    else:
        ITix += 1
        
IT_Aspects        = []
IT_Description    = []
IT_Dimension      = []
IT_Classification = []
IT_Selector       = []
IT_IndexLetter    = []
ITix += 2 # start on first data row
while True:
    if len(Model_Configsheet.cell_value(ITix,2)) > 0:
        IT_Aspects.append(Model_Configsheet.cell_value(ITix,2))
        IT_Description.append(Model_Configsheet.cell_value(ITix,3))
        IT_Dimension.append(Model_Configsheet.cell_value(ITix,4))
        IT_Classification.append(Model_Configsheet.cell_value(ITix,5))
        IT_Selector.append(Model_Configsheet.cell_value(ITix,6))
        IT_IndexLetter.append(Model_Configsheet.cell_value(ITix,7))        
        ITix += 1
    else:
        break

print('Read parameter list from model config sheet.')
PLix = 0
while True: # search for parameter list entry
    if Model_Configsheet.cell_value(PLix,1) == 'Model Parameters':
        break
    else:
        PLix += 1
        
PL_Names          = []
PL_Description    = []
PL_Version        = []
PL_IndexStructure = []
PL_IndexMatch     = []
PL_IndexLayer     = []
PLix += 2 # start on first data row
while True:
    if len(Model_Configsheet.cell_value(PLix,2)) > 0:
        PL_Names.append(Model_Configsheet.cell_value(PLix,2))
        PL_Description.append(Model_Configsheet.cell_value(PLix,3))
        PL_Version.append(Model_Configsheet.cell_value(PLix,4))
        PL_IndexStructure.append(Model_Configsheet.cell_value(PLix,5))
        PL_IndexMatch.append(Model_Configsheet.cell_value(PLix,6))
        # strip numbers out of list string
        PL_IndexLayer.append(msf.ListStringToListNumbers(Model_Configsheet.cell_value(PLix,7))) 
        PLix += 1
    else:
        break
    
  

print('Read model run control from model config sheet.')
PrLix = 0
while True: # search for model flow control entry
    if Model_Configsheet.cell_value(PrLix,1) == 'Model flow control':
        break
    else:
        PrLix += 1
        
PrLix += 2 # start on first data row
while True:
    if Model_Configsheet.cell_value(PrLix,2) != '':
        try:
            ScriptConfig[Model_Configsheet.cell_value(PrLix,2)] = Model_Configsheet.cell_value(PrLix,3)
        except:
            None
        PrLix += 1
    else:
        break
    
    
print('Define model classifications and select items for model classifications according to information provided by config file.')
ModelClassification  = {} # Dict of model classifications
for m in range(0,len(IT_Aspects)):
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



print('Read process list from model config sheet.')
PrLix = 0
while True: # search for process list entry
    if Model_Configsheet.cell_value(PrLix,1) == 'Process Group List':
        break
    else:
        PrLix += 1
        
PrL_Number         = []
PrL_Name           = []
PrL_Code           = []
PrL_Type           = []
PrLix += 2 # start on first data row
while True:
    if Model_Configsheet.cell_value(PrLix,2) != '':
        try:
            PrL_Number.append(int(Model_Configsheet.cell_value(PrLix,2)))
        except:
            PrL_Number.append(Model_Configsheet.cell_value(PrLix,2))
        PrL_Name.append(Model_Configsheet.cell_value(PrLix,3))
        PrL_Code.append(Model_Configsheet.cell_value(PrLix,4))
        PrL_Type.append(Model_Configsheet.cell_value(PrLix,5))
        PrLix += 1
    else:
        break  




PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
                      Geogr_Scope = 'World', 
                      Unit = 'Mt', 
                      ProcessList = [], 
                      FlowDict = mfa_system.FlowDict, 
                      StockDict = mfa_system.StockDict,
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
