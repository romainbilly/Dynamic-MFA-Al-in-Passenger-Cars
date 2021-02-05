# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:14:09 2021

@author: romainb
"""
import ODYM_Classes as msc # import the ODYM class file

# PassengerVehicleFleet_MFA_System = msc.MFAsystem(Name = 'Global_Passengers_Vehicle_Fleet', 
#                       Geogr_Scope = 'World', 
#                       Unit = 'Mt', 
#                       ProcessList = [], 
#                       FlowDict = {}, 
#                       StockDict = {},
#                       ParameterDict = ParameterDict, 
#                       Time_Start = Model_Time_Start, 
#                       Time_End = Model_Time_End, 
#                       IndexTable = IndexTable, 
#                       Elements = IndexTable.loc['Element'].Classification.Items, 
#                       Graphical = None) # Initialize MFA system
                      
# # Check Validity of index tables:
# # returns true if dimensions are OK and time index is present and element list is not empty
# IndexTableCheck() 



# # Add processes to system
# for m in range(0, len(PrL_Number)):
#     ProcessList.append(msc.Process(Name = PrL_Name[m], ID   = PrL_Number[m]))
  

# Define system variables: 16 flows.
FlowDict = {}  
FlowDict['F_0_1'] = msc.Flow(Name = 'Primary Aluminium demand', P_Start = 0,
                                                  P_End = 1, Indices = 't,e,a,S',
                                                  Values=None, Uncert=None, Color = None,
                                                  ID = None, UUID = None)     
FlowDict['F_1_2'] = msc.Flow(Name = 'Materials for Passenger vehicle production', P_Start = 1,
                                                  P_End = 2, Indices = 't,r,e,a,S',
                                                  Values=None, Uncert=None, Color = None,
                                                  ID = None, UUID = None)
FlowDict['F_1_9'] = msc.Flow(Name = 'Scrap surplus', P_Start = 1, 
                                                  P_End = 9, Indices = 't,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_2_3'] = msc.Flow(Name = 'New registration of vehicles', P_Start = 2, 
                                                  P_End = 3, Indices = 't,r,p,s,z,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_3_4'] = msc.Flow(Name = 'End of Life vehicles', P_Start = 3, 
                                                  P_End = 4, Indices = 't,c,r,p,s,z,e,a,S',
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_4_0'] = msc.Flow(Name = 'Collection losses', P_Start = 4, 
                                                  P_End = 0, Indices = 't,c,r,p,s,z,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_4_5'] = msc.Flow(Name = 'Collected cars to dismantling', P_Start = 4, 
                                                  P_End = 5, Indices = 't,c,r,p,s,z,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_4_7'] = msc.Flow(Name = 'Collected cars directly to shredding', P_Start = 4, 
                                                  P_End = 7, Indices = 't,c,r,p,s,z,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_5_6'] = msc.Flow(Name = 'Dismantled components to shredding', P_Start = 5, 
                                                  P_End = 6, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_5_7'] = msc.Flow(Name = 'Residues from dismantllng to shredding', P_Start = 5, 
                                                  P_End = 7, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_6_0'] = msc.Flow(Name = 'Shredding losses', P_Start = 6, 
                                                  P_End = 0, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_6_1'] = msc.Flow(Name = 'Al scrap from dismantled components', P_Start = 6, 
                                                  P_End = 1, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_7_0'] = msc.Flow(Name = 'Shredding losses', P_Start = 7, 
                                                  P_End = 0, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_7_1'] = msc.Flow(Name = 'Mixed Al scrap', P_Start = 7, 
                                                  P_End = 1, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)
FlowDict['F_7_8'] = msc.Flow(Name = 'Mixed Al scrap to alloy sorting', P_Start = 7, 
                                                  P_End = 8, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)          
FlowDict['F_8_1'] = msc.Flow(Name = 'Alloy sorted scrap', P_Start = 8, 
                                                  P_End = 1, Indices = 't,r,e,a,S', 
                                                  Values=None, Uncert=None, Color = None, 
                                                  ID = None, UUID = None)                                               
                                                                                       
                                                  
# Define system variables: 1 stock and 1 stock change:
StockDict = {}
StockDict['S_3']  = msc.Stock(Name = 'In-use stock', P_Res = 3, Type = 0,
                                                  Indices = 't,c,r,p,s,z,e,a,S', Values=None, Uncert=None,
                                                  ID = None, UUID = None)

StockDict['dS_3']  = msc.Stock(Name = 'Net in-use stock change', P_Res = 3, Type = 1,
                                                  Indices = 't,r,p,s,z,e,a,S', Values=None, Uncert=None,
                                                  ID = None, UUID = None)

# Initialize_StockValues() # Assign empty arrays to stocks according to dimensions.
# Initialize_FlowValues() # Assign empty arrays to flows according to dimensions. 
