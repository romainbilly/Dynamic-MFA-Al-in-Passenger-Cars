# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:31:59 2020

@author: romainb
"""

# Load a local copy of the current ODYM branch:
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xlrd, xlwt
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
import mfa_system as mss # import the system definition

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
SCix = 0
# search for script config list entry
while Model_Configsheet.cell_value(SCix, 1) != 'General Info':
    SCix += 1
        
SCix += 2  # start on first data row
while len(Model_Configsheet.cell_value(SCix, 3)) > 0:
    ScriptConfig[Model_Configsheet.cell_value(SCix, 2)] = Model_Configsheet.cell_value(SCix,3)
    SCix += 1

SCix = 0
# search for script config list entry
while Model_Configsheet.cell_value(SCix, 1) != 'Software version selection':
    SCix += 1
        
SCix += 2 # start on first data row
while len(Model_Configsheet.cell_value(SCix, 3)) > 0:
    ScriptConfig[Model_Configsheet.cell_value(SCix, 2)] = Model_Configsheet.cell_value(SCix,3)
    SCix += 1 
    
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
print('Read model data and parameters.')

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
    
    
Mylog.info('### 4 - Define MFA system')
print('Define MFA system and processes.')

PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
                      Geogr_Scope = 'World', 
                      Unit = 'Mt', 
                      ProcessList = [], 
                      FlowDict = mss.FlowDict, 
                      StockDict = mss.StockDict,
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
    
# # Define system variables: 6 flows.
# PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'] = msc.Flow(Name = 'Primary Aluminium demand', P_Start = 0,
#                                                   P_End = 1, Indices = 't,e,a,S',
#                                                   Values=None, Uncert=None, Color = None,
#                                                   ID = None, UUID = None)     
# PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'] = msc.Flow(Name = 'Materials for Passenger vehicle production', P_Start = 1,
#                                                   P_End = 2, Indices = 't,r,e,a,S',
#                                                   Values=None, Uncert=None, Color = None,
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'] = msc.Flow(Name = 'Scrap surplus', P_Start = 1, 
#                                                   P_End = 9, Indices = 't,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'] = msc.Flow(Name = 'New registration of vehicles', P_Start = 2, 
#                                                   P_End = 3, Indices = 't,r,p,s,z,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'] = msc.Flow(Name = 'End of Life vehicles', P_Start = 3, 
#                                                   P_End = 4, Indices = 't,c,r,p,s,z,e,a,S',
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'] = msc.Flow(Name = 'Collection losses', P_Start = 4, 
#                                                   P_End = 0, Indices = 't,c,r,p,s,z,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'] = msc.Flow(Name = 'Collected cars to dismantling', P_Start = 4, 
#                                                   P_End = 5, Indices = 't,c,r,p,s,z,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'] = msc.Flow(Name = 'Collected cars directly to shredding', P_Start = 4, 
#                                                   P_End = 7, Indices = 't,c,r,p,s,z,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'] = msc.Flow(Name = 'Dismantled components to shredding', P_Start = 5, 
#                                                   P_End = 6, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'] = msc.Flow(Name = 'Residues from dismantllng to shredding', P_Start = 5, 
#                                                   P_End = 7, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'] = msc.Flow(Name = 'Shredding losses', P_Start = 6, 
#                                                   P_End = 0, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'] = msc.Flow(Name = 'Al scrap from dismantled components', P_Start = 6, 
#                                                   P_End = 1, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'] = msc.Flow(Name = 'Shredding losses', P_Start = 7, 
#                                                   P_End = 0, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'] = msc.Flow(Name = 'Mixed Al scrap', P_Start = 7, 
#                                                   P_End = 1, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)
# PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'] = msc.Flow(Name = 'Mixed Al scrap to alloy sorting', P_Start = 7, 
#                                                   P_End = 8, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)          
# PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'] = msc.Flow(Name = 'Alloy sorted scrap', P_Start = 8, 
#                                                   P_End = 1, Indices = 't,r,e,a,S', 
#                                                   Values=None, Uncert=None, Color = None, 
#                                                   ID = None, UUID = None)                                               
                                                                                       
                                                  
# # Define system variables: 1 stock and 1 stock change:
# PassengerVehicleFleet_MFA_System.StockDict['S_3']  = msc.Stock(Name = 'In-use stock', P_Res = 3, Type = 0,
#                                                   Indices = 't,c,r,p,s,z,e,a,S', Values=None, Uncert=None,
#                                                   ID = None, UUID = None)

# PassengerVehicleFleet_MFA_System.StockDict['dS_3']  = msc.Stock(Name = 'Net in-use stock change', P_Res = 3, Type = 1,
#                                                   Indices = 't,r,p,s,z,e,a,S', Values=None, Uncert=None,
#                                                   ID = None, UUID = None)

PassengerVehicleFleet_MFA_System.Initialize_StockValues() # Assign empty arrays to stocks according to dimensions.
PassengerVehicleFleet_MFA_System.Initialize_FlowValues() # Assign empty arrays to flows according to dimensions. 


Mylog.info('### 5 - Building and solving the MFA model')
# 1) Determine vehicle inflow and outflow by age-cohort from stock and lifetime data. 
# These calculations are done outside of the MFA system 
# as we are not yet on the material level but at the product level.

O_tcr = np.zeros((Nt,Nt,Nr))
S_tcr= np.zeros((Nt,Nt,Nr))
DS_tr = np.zeros((Nt,Nr))
I_cr = np.zeros((Nt,Nr))
O_tr = np.zeros((Nt,Nr))

O_tcrS = np.zeros((Nt,Nt,Nr,NS))
S_tcrS= np.zeros((Nt,Nt,Nr,NS))
DS_trS = np.zeros((Nt,Nr,NS))
I_crS = np.zeros((Nt,Nr,NS))
O_trS = np.zeros((Nt,Nr,NS))



for scenario in range(NS):
    print('Computing Scenario:  ',IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('S')].Items[scenario])
    
    print('Solving dynamic stock model of the passenger vehicle fleet for: ')
    for region in range(Nr):
             
        # 1a) Loop over all regions to determine inflow-driven stock of vehicles, with pre 2005 age-cohorts absent
        print(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('r')].Items[region])
        # Create helper DSM for computing the dynamic stock model:
        DSM = dsm.DynamicStockModel(t = np.array(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items),
                                           s = PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Stock'].Values[region,:], 
                                           lt = {'Type': 'Normal', 'Mean': PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Lifetime'].Values[:,region],
                                                 'StdDev': PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Lifetime'].Values[:,region]/4} )
        
        Stock_by_cohort = DSM.compute_stock_driven_model()
    
    
    
        #print(Stock_by_cohort.shape)
        O_tcr[:,:,region] = DSM.compute_o_c_from_s_c()
        O_tr[:,region] = DSM.compute_outflow_total()
        S_tcr[:,:,region] = DSM.s_c
        I_cr[:,region] = DSM.i
        DS_tr[:,region] = DSM.compute_stock_change()
        S_tr = np.einsum('tcr -> tr', S_tcr)
        
        
        O_tcrS[:,:,region,scenario] = DSM.compute_o_c_from_s_c()
        O_trS[:,region,scenario] = DSM.compute_outflow_total()
        S_tcrS[:,:,region,scenario] = DSM.s_c
        I_crS[:,region,scenario] = DSM.i
        DS_trS[:,region,scenario] = DSM.compute_stock_change()
        S_trS = np.einsum('tcrS -> trS', S_tcrS)



#Stock py powertrain and segment with scenarios
print("Performing Stock calculations")
S_tcrpS = np.einsum('tcrS, Srpc -> tcrpS', S_tcrS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values) 
S_tcrpsS = np.einsum('tcrpS, Srsc -> tcrpsS', S_tcrpS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values) 

S_trpsS = np.einsum('tcrpsS -> trpsS', S_tcrpsS) 
S_tpsS = np.einsum('tcrpsS -> tpsS', S_tcrpsS) 
S_tpS = np.einsum('tcrpsS -> tpS', S_tcrpsS) 
S_tsS = np.einsum('tcrpsS -> tsS', S_tcrpsS) 

I_crpS = np.einsum('cr, Srpc -> crpS', I_cr, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values) 
I_crpsS = np.einsum('crpS, Srsc -> crpsS', I_crpS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values) 
I_csS = np.einsum('crpsS -> csS', I_crpsS) 
I_cpS = np.einsum('crpsS -> cpS', I_crpsS) 


O_tcrpS = np.einsum('tcr, Srpc -> tcrpS', O_tcr, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values) 
O_tcrpsS = np.einsum('tcrpS, Srsc -> tcrpsS', O_tcrpS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values) 
O_tsS = np.einsum('tcrpsS -> tsS', O_tcrpsS) 
O_tpS = np.einsum('tcrpsS -> tpS', O_tcrpsS) 

#Aluminium content calculations by scenario
print("Performing Al content calculations")
# Stock
Al_stock_tcrpsS = np.einsum('tcrpsS, erc -> tcrpsS', S_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values) 
Al_stock_tcrpsS_pseg = np.einsum('tcrpsS, sc -> tcrpsS', Al_stock_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values) 
Al_stock_tcrpsS_pseg_ptype = np.einsum('tcrpsS, pc -> tcrpsS', Al_stock_tcrpsS_pseg, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values) 
Al_stock_tcrpsS = Al_stock_tcrpsS_pseg_ptype

Al_stock_trS = np.einsum('tcrpsS -> trS', Al_stock_tcrpsS_pseg_ptype)

# Inflow
Al_inflow_crpsS = np.einsum('crpsS, erc -> crpsS', I_crpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values) 
Al_inflow_crpsS_pseg = np.einsum('crpsS, sc -> crpsS', Al_inflow_crpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values) 
Al_inflow_crpsS_pseg_ptype = np.einsum('crpsS, pc -> crpsS', Al_inflow_crpsS_pseg, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values) 
Al_inflow_crpsS = Al_inflow_crpsS_pseg_ptype
Al_inflow_crS = np.einsum('crpsS-> crS ', Al_inflow_crpsS) 


# Outflow
Al_outflow_tcrpsS = np.einsum('tcrpsS, erc -> tcrpsS', O_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values) 
Al_outflow_tcrpsS_pseg = np.einsum('tcrpsS, sc -> tcrpsS', Al_outflow_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values) 
Al_outflow_tcrpsS_pseg_ptype = np.einsum('tcrpsS, pc -> tcrpsS', Al_outflow_tcrpsS_pseg, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values) 
Al_outflow_tcrpsS = Al_outflow_tcrpsS_pseg_ptype
Al_outflow_trS = np.einsum('tcrpsS -> trS', Al_outflow_tcrpsS) 


# Component level by Scenario
print("Performing component level calculations")
Components = IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('z')].Items

Al_stock_tcrpszS = np.einsum('tcrpsS, crpsz -> tcrpszS', Al_stock_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values) 
Al_stock_tzS = np.einsum('tcrpszS -> tzS ', Al_stock_tcrpszS)

Al_inflow_crpszS = np.einsum('crpsS, crpsz -> crpszS ', Al_inflow_crpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values) 
Al_inflow_czS = np.einsum('crpszS -> czS', Al_inflow_crpszS)

Al_outflow_tcrpszS = np.einsum('tcrpsS, crpsz -> tcrpszS', Al_outflow_tcrpsS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values) 
Al_outflow_tzS = np.einsum('tcrpszS -> tzS', Al_outflow_tcrpszS)



# Aluminium Alloys calculation
print("Performing alloy content calculations")
Alloys = IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('a')].Items

Alloys_inflow_crpszaS = np.einsum('crpszS, az -> crpszaS', Al_inflow_crpszS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values) 

Alloys_inflow_craS = np.einsum('crpszaS -> craS', Alloys_inflow_crpszaS)
Alloys_inflow_caS = np.einsum('craS-> caS', Alloys_inflow_craS)

Alloys_outflow_tcrpszaS = np.einsum('tcrpszS, az -> tcrpszaS', Al_outflow_tcrpszS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values) 
Alloys_outflow_traS = np.einsum('tcrpszaS -> traS', Alloys_outflow_tcrpszaS)
Alloys_outflow_taS = np.einsum('traS -> taS ', Alloys_outflow_traS)

Alloys_stock_tcrpszaS = np.einsum('tcrpszS, az -> tcrpszaS', Al_stock_tcrpszS, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values) 




# Solving the MFA system
print("Solving the MFA system")
# F_2_3, dimensions trpszea
PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values = np.expand_dims(Alloys_inflow_crpszaS, axis=5)
# F_1_2, Materials for Passenger vehicle production
PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values = \
        np.einsum('trpszeaS-> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values) 
# F_3_4, EoL Vehicles, dimensions tcrpszea
PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values = np.expand_dims(Alloys_outflow_tcrpszaS, axis=6)
# F_4_0, dimensions tcrpszea
PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values = np.einsum('tcrpszeaS, tr -> tcrpszeaS',
                                         PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values,
                                         1 - PassengerVehicleFleet_MFA_System.ParameterDict['Collection'].Values)
# F_4_5, Collected cars to dismantling, dimensions tcrpszeaS
PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values = np.einsum('tcrpszeaS, rzt -> tcrpszeaS',
                                         PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values - \
                                         PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values,
                                         PassengerVehicleFleet_MFA_System.ParameterDict['Dismantling'].Values)
# F_4_7, Collected cars to shredding, dimensions tcrpszea
PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values = \
        PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values
# F_5_6, Dismantled components to shredding, dimensions trea
# need to add dismantling yield
PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values = \
        0.7 * np.einsum('tcrpszeaS-> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values) 

                                         
# F_5_7, Residues from dismantllng to shredding
# need to add dismantling yield
PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values = \
        0.3 * np.einsum('tcrpszeaS -> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values) 


# F_6_0, Shredding losses, dimensions trea
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values = \
        0.05 * PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values

# F_6_1, Al scrap from dismantled components, dimensions trea
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values = \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values

# F_7_0, Shredding losses, dimensions trea
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values =  0.05 * (
        np.einsum('tcrpszeaS-> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values)

# F_7_8, Scrap to alloy sorting, dimensions trea
PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values = \
np.einsum('treaS, tr -> treaS',
        np.einsum('tcrpszeaS-> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values,
        PassengerVehicleFleet_MFA_System.ParameterDict['Alloy_Sorting'].Values)

# F_7_1, Mixed shredded scrap, dimensions trea
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values = \
        np.einsum('tcrpszeaS-> treaS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values
# Alloy composition adjusted to become secondary castings only
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[:,:,:,2] =\
        np.einsum('treaS -> treS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values)  
# Setting wrought and primary casting to zero        
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values = \
        np.einsum('treaS, a -> treaS', 
                  PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values, 
                  np.array([0,0,1]))

# F_8_1, Alloy sorted scrap, dimensions trea
PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values = PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values

               
# Correcting for scrap surplus
# Scrap surplus considered at global level only
print("Correcting for scrap surplus")
# Mass balance of process 1 without scrap surplus and primary production
# If positive, there is a scrap surplus for the alloy considered
Process_1_mb_teaS = np.einsum('treaS-> teaS', 
        PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values)
scrap_surplus_teaS = np.zeros(Process_1_mb_teaS.shape)        

for it,ie,ia,iS in np.ndindex(Process_1_mb_teaS.shape):
    if Process_1_mb_teaS[it,ie,ia,iS] > 0:
        scrap_surplus_teaS[it,ie,ia,iS] = Process_1_mb_teaS[it,ie,ia,iS]



PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values = scrap_surplus_teaS       
scrap_surplus_taS =  np.einsum('teaS-> taS', scrap_surplus_teaS)


# F_0_1, Primary Aluminium Demand, determined by mass balance
PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values = \
        np.einsum('treaS-> teaS', 
        PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values     

        
# S_3, dimensions tcrpsza
PassengerVehicleFleet_MFA_System.StockDict['S_3'].Values = np.expand_dims(Alloys_stock_tcrpszaS, axis=3)
# dS_3, dimensions trpsza
PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values = PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values - \
                                                            np.einsum('tcrpszeaS-> trpszeaS', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values)
                                                            
                                                            
                                                            
#### Carbon footprint calculations   
carbon_footprint_primary = np.einsum('teaS, tS -> tS', 
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values,
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Primary'].Values)
carbon_footprint_secondary = np.einsum('teaS, tS -> tS', 
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values - \
                                     np.einsum('treaS -> teaS', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values),
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Secondary'].Values)

                                                         
                                                           
# Mass balance check:
print("Checking Mass Balance")    
Bal = PassengerVehicleFleet_MFA_System.MassBalance()
print(np.abs(Bal).sum(axis = 0)) # reports the sum of all absolute balancing errors by process.        

# Exports
print("Exporting data")
F_1_2_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values)
F_2_3_taS = np.einsum('trpszeaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values)
F_3_4_taS = np.einsum('tcrpszeaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values)
F_4_0_taS = np.einsum('tcrpszeaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values)
F_4_5_taS = np.einsum('tcrpszeaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values)
F_4_7_taS = np.einsum('tcrpszeaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values)
F_5_6_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values)
F_5_7_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values)
F_6_0_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values)
F_6_1_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values)
F_7_0_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values)
F_7_1_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values)
F_7_8_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values)
F_8_1_taS = np.einsum('treaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values)
F_0_1_taS = np.einsum('teaS -> taS', PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values)
F_1_9_taS = scrap_surplus_taS
S_3_taS = np.einsum('tcrpszeaS -> taS', PassengerVehicleFleet_MFA_System.StockDict['S_3'].Values)
dS_3_taS = np.einsum('trpszeaS -> taS', PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values)


cf.export_to_csv(F_2_3_taS, 'F_2_3_taS', IndexTable)
cf.export_to_csv(F_3_4_taS, 'F_3_4_taS', IndexTable)
cf.export_to_csv(F_4_0_taS, 'F_4_0_taS', IndexTable)
cf.export_to_csv(F_4_5_taS, 'F_4_5_taS', IndexTable)
cf.export_to_csv(F_4_7_taS, 'F_4_7_taS', IndexTable)
cf.export_to_csv(F_5_6_taS, 'F_5_6_taS', IndexTable)
cf.export_to_csv(F_5_7_taS, 'F_5_7_taS', IndexTable)
cf.export_to_csv(F_6_0_taS, 'F_6_0_taS', IndexTable)
cf.export_to_csv(F_6_1_taS, 'F_6_1_taS', IndexTable)
cf.export_to_csv(F_7_0_taS, 'F_7_0_taS', IndexTable)
cf.export_to_csv(F_7_1_taS, 'F_7_1_taS', IndexTable)
cf.export_to_csv(F_1_2_taS, 'F_1_2_taS', IndexTable)
cf.export_to_csv(F_1_9_taS, 'scrap_surplus_taS', IndexTable)

iterables = []
names = []
for dim in ['t','a','S']:
    iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
    names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  

index = pd.MultiIndex.from_product(iterables, names=names)
df = pd.DataFrame(F_2_3_taS.flatten()/10**9,index=index, columns = ['F_2_3_ta'])
df['F_3_4_ta'] = F_3_4_taS.flatten()/10**9
df['F_4_0_ta'] = F_4_0_taS.flatten()/10**9
df['F_4_5_ta'] = F_4_5_taS.flatten()/10**9
df['F_4_7_ta'] = F_4_7_taS.flatten()/10**9
df['F_5_6_ta'] = F_5_6_taS.flatten()/10**9
df['F_5_7_ta'] = F_5_7_taS.flatten()/10**9
df['F_6_0_ta'] = F_6_0_taS.flatten()/10**9
df['F_6_1_ta'] = F_6_1_taS.flatten()/10**9
df['F_7_0_ta'] = F_7_0_taS.flatten()/10**9
df['F_7_1_ta'] = F_7_1_taS.flatten()/10**9
df['F_1_2_ta'] = F_1_2_taS.flatten()/10**9
df['F_1_9_ta'] = scrap_surplus_taS.flatten()/10**9
df['S_3_ta'] = S_3_taS.flatten()/10**9
df['dS_3_ta'] = dS_3_taS.flatten()/10**9
df['F_0_1_ta'] = F_0_1_taS.flatten()/10**9
df['F_7_8_ta'] = F_7_8_taS.flatten()/10**9
df['F_8_1_ta'] = F_8_1_taS.flatten()/10**9



df.to_excel('results/flows_scenarios.xlsx', merge_cells=False)

# F_0_1_t = np.einsum('ta -> t', F_0_1_ta)/10**9
# F_1_2_t = np.einsum('ta -> t', F_1_2_ta)/10**9
# F_1_9_t = np.einsum('ta -> t', F_1_9_ta)/10**9
# F_2_3_t = np.einsum('ta -> t', F_2_3_ta)/10**9
# F_3_4_t = np.einsum('ta -> t', F_3_4_ta)/10**9
# F_4_0_t = np.einsum('ta -> t', F_4_0_ta)/10**9
# F_4_5_t = np.einsum('ta -> t', F_4_5_ta)/10**9
# F_4_7_t = np.einsum('ta -> t', F_4_7_ta)/10**9
# F_5_6_t = np.einsum('ta -> t', F_5_6_ta)/10**9
# F_5_7_t = np.einsum('ta -> t', F_5_7_ta)/10**9
# F_6_0_t = np.einsum('ta -> t', F_6_0_ta)/10**9
# F_6_1_t = np.einsum('ta -> t', F_6_1_ta)/10**9
# F_7_0_t = np.einsum('ta -> t', F_7_0_ta)/10**9
# F_7_1_t = np.einsum('ta -> t', F_7_1_ta)/10**9
# F_7_8_t = np.einsum('ta -> t', F_7_8_ta)/10**9
# F_8_1_t = np.einsum('ta -> t', F_8_1_ta)/10**9




# index = pd.Index(
#         PassengerVehicleFleet_MFA_System.IndexTable['Classification']['Time'].Items[:],
#         name="Time")

# df = pd.DataFrame(F_0_1_t.flatten(),index=index, columns = ['F_0_1_t'])

# df['F_1_2_t'] = F_1_2_t.flatten()
# df['F_1_9_t'] = F_1_9_t.flatten()
# df['F_2_3_t'] = F_2_3_t.flatten()
# df['F_3_4_t'] = F_3_4_t.flatten()
# df['F_4_0_t'] = F_4_0_t.flatten()
# df['F_4_5_t'] = F_4_5_t.flatten()
# df['F_4_7_t'] = F_4_7_t.flatten()
# df['F_5_6_t'] = F_5_6_t.flatten()
# df['F_5_7_t'] = F_5_7_t.flatten()
# df['F_6_0_t'] = F_6_0_t.flatten()
# df['F_6_1_t'] = F_6_1_t.flatten()
# df['F_7_0_t'] = F_7_0_t.flatten()
# df['F_7_1_t'] = F_7_1_t.flatten()
# df['F_7_8_t'] = F_7_8_t.flatten()
# df['F_8_1_t'] = F_8_1_t.flatten()



# df.to_excel('results/flows_per_year.xlsx')


#### Plots
print("Plots")
plt.ioff() 

for scenario in range(NS):

    print("Plotting results for scenario " + str(scenario))
    ## Car Stock per region
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Region',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_trS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows per region
    y_dict = {
            'name': 'Car Inflows',
            'aspect': 'Region',
            'unit': 'cars/yr'
            }
    cf.plot_result_time_scenario(I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    # Car outflows per region
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Region',
            'unit': 'cars/yr'
            }
    cf.plot_result_time_scenario(O_trS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Stock by powertrain
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows by powertrain
    y_dict = {
            'name': 'Car inflows',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Car outflows by powertrain
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(O_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Stock by segment
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows by segment
    y_dict = {
            'name': 'Car Inflows',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Car outflows by segment
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(O_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    
    
    ## Plot Al Stock per component
    y_dict = {
            'name': 'Al Stock',
            'aspect': 'Component',
            'unit': 'Mt'
            }
    cf.plot_result_time_scenario(Al_stock_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Plot Al inflows per conponent
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Component',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_inflow_czS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    
    ## Plot Al outflows per conponent
    y_dict = {
            'name': 'Al Outflows',
            'aspect': 'Component',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_outflow_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    
    
    # Aluminium stock per region
    y_dict = {
            'name': 'Al stock',
            'aspect': 'Region',
            'unit': 'Mt'
            }
    cf.plot_result_time_scenario(Al_stock_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    # Aluminium inflows per region
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Region',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_inflow_crS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    # Aluminium outflows per region
    y_dict = {
            'name': 'Al Outflows',
            'aspect': 'Region',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_outflow_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Plot Al Alloys inflows 
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Alloy',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Alloys_inflow_caS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    ## Plot Average Al content in inflows per region
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Region',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_crS / I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='no')
    
    
    ## Plot Average Al content in inflows per powertrain
    I_cpS = np.einsum('crpsS -> cpS', I_crpsS)
    Al_inflow_cpS = np.einsum('crpsS -> cpS', Al_inflow_crpsS)
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Powertrain',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_cpS / I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='no')
    
    
    ## Plot Average Al content in inflows per segment
    I_csS = np.einsum('crpsS -> csS', I_crpsS)
    Al_inflow_csS = np.einsum('crpsS -> csS', Al_inflow_crpsS)
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Segment',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_csS / I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='no')
    
    ## Plot Al Alloys outflows
    y_dict = {
            'name': 'Al Outflows',
            'aspect': 'Alloy',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Alloys_outflow_taS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no', stack='yes')
    
    # Plot Ratio Outflows / Inflows
    y_dict = {
            'name': 'Ratio Outflows - Inflows',
            'aspect': 'Alloy',
            'unit': ''
            }
    cf.plot_result_time_scenario(Alloys_outflow_taS / Alloys_inflow_caS, y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no')
    
    # Plot Ratio Outflows / Inflows for secondary castings per region
    y_dict = {
            'name': 'O-I Ratio for 2nd castings',
            'aspect': 'Region',
            'unit': ''
            }
    cf.plot_result_time_scenario(Alloys_outflow_traS[:,:,2] / Alloys_inflow_craS[:,:,2], y_dict, IndexTable, t_min= 100, t_max = 151, scenario=scenario, show = 'no')


## Plot Total Carbon footprint
y_dict = {
        'name': 'Carbon footprint of Al production',
        'aspect': 'Scenario',
        'unit': 'kg CO2'
        }
cf.plot_result_time(carbon_footprint_primary + carbon_footprint_secondary, y_dict, IndexTable, t_min= 100, t_max = 151, show = 'no', stack='no')
    
