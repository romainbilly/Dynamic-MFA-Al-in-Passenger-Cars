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
import time
from datetime import datetime


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
from data_prep_class import DataPrep

# Initialize loggin routine
log_verbosity = eval("log.DEBUG")
log_filename = 'LogFileTest.md'
log.getLogger('matplotlib').setLevel(log.WARNING)
[Mylog, console_log, file_log] = msf.function_logger(log_filename, os.getcwd(),
                                                     log_verbosity, log_verbosity)
Mylog.info('### 1. - Initialize.')
Data_Prep = DataPrep(DataPath,Mylog)      
# Data_Prep = DataPrep(DataPath,Mylog, "ParameterDict.p")


# Mylog.info('### 3 - Read classification and data')


IndexTable = Data_Prep.IndexTable
#Define shortcuts for the most important index sizes:
Nt = len(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)
Nr = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('r')].Items)
Np = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('p')].Items) 
Ns = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('s')].Items) 
Nz = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('z')].Items)
Na = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('a')].Items)
NS = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('S')].Items)


# print('Read model data and parameters.')
# ParameterDict = data_prep.get_parameter_dict(data_prep.DataPath, data_prep.PL_Names)


Mylog.info('### 4 - Define MFA system')
print('Define MFA system and processes.')

# PassengerVehicleFleet_MFA_System = data_prep.PassengerVehicleFleet_MFA_System
PassengerVehicleFleet_MFA_System = Data_Prep.create_mfa_system()
                      

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


print('Solving dynamic stock model of the passenger vehicle fleet')
for scenario in range(NS):
    for region in range(Nr):
        # 1a) Loop over all regions to determine a stock-driven model of the global passenger vehicle fleet
        # Create helper DSM for computing the dynamic stock model:
        DSM = dsm.DynamicStockModel(t = np.array(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items),
                                           s = PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Stock'].Values[scenario,region,:], 
                                           lt = {'Type': 'Normal', 'Mean': PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Lifetime'].Values[:,region],
                                                 'StdDev': PassengerVehicleFleet_MFA_System.ParameterDict['Vehicle_Lifetime'].Values[:,region]/4} )
        
        Stock_by_cohort = DSM.compute_stock_driven_model()
    
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




print("Performing Stock calculations")
# Calculating segment split by powertrain 
Powertrain_Srpc = PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values
Segment_Srsc = PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values
PS_Srpsc =  np.einsum('Srpc, Srsc -> Srpsc', Powertrain_Srpc, Segment_Srsc)

# Correction according to SP_Coeff parameter
PS_Srpsc =  np.einsum('Srpsc, psc -> Srpsc', PS_Srpsc,
                      PassengerVehicleFleet_MFA_System.ParameterDict['SP_Coeff'].Values)
# for powertrains HEV, PHEV, and BEV, a correction coefficient is applied to the segments AB, DE, and SUV. 
# Segment C is calculated by "mass balance" to reach the average powertrain split 
PS_Srpsc[:,:,1,1,:] = Powertrain_Srpc[:,:,1,:] - PS_Srpsc[:,:,1,0,:] - PS_Srpsc[:,:,1,2,:] - PS_Srpsc[:,:,1,3,:]
PS_Srpsc[:,:,2,1,:] = Powertrain_Srpc[:,:,2,:] - PS_Srpsc[:,:,2,0,:] - PS_Srpsc[:,:,2,2,:] - PS_Srpsc[:,:,2,3,:]
PS_Srpsc[:,:,3,1,:] = Powertrain_Srpc[:,:,3,:] - PS_Srpsc[:,:,3,0,:] - PS_Srpsc[:,:,3,2,:] - PS_Srpsc[:,:,3,3,:]
# the segment split of the ICEV segment is calculated from the other powertrain types to reach the average segment split
for s in range(Ns):
    PS_Srpsc[:,:,0,s,:] = Segment_Srsc[:,:,s,:] -  np.sum(PS_Srpsc[:,:,1:,s,:], axis=2)


# Stock py powertrain and segment with scenarios
S_tcrpS = np.einsum('tcrS, Srpc -> tcrpS', S_tcrS, Powertrain_Srpc) 
S_tcrsS = np.einsum('tcrS, Srsc -> tcrsS', S_tcrS, Segment_Srsc) 
S_tcrpsS= np.einsum('tcrS, Srpsc -> tcrpsS', S_tcrS, PS_Srpsc) 
S_tpsS = np.einsum('tcrpsS -> tpsS', S_tcrpsS) 
S_tpS = np.einsum('tcrpsS -> tpS', S_tcrpsS) 
S_tsS = np.einsum('tcrpsS -> tsS', S_tcrpsS) 
S_tS = np.einsum('tsS -> tS', S_tsS) 

I_crpS = np.einsum('crS, Srpc -> crpS', I_crS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values) 
I_crsS = np.einsum('crS, Srsc -> crS', I_crS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values) 
I_crpsS= np.einsum('crS, Srpsc -> crpsS', I_crS, PS_Srpsc) 
I_csS = np.einsum('crpsS -> csS', I_crpsS) 
I_cpS = np.einsum('crpsS -> cpS', I_crpsS) 


O_tcrpS = np.einsum('tcrS, Srpc -> tcrpS', O_tcrS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values) 
O_tcrsS = np.einsum('tcrS, Srsc -> tcrsS', O_tcrS, 
                    PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values) 
O_tcrpsS = np.einsum('tcrS, Srpsc -> tcrpsS', O_tcrS, PS_Srpsc) 
O_tsS = np.einsum('tcrpsS -> tsS', O_tcrpsS) 
O_tpS = np.einsum('tcrpsS -> tpS', O_tcrpsS) 


# I_crps_corr = np.einsum('cr, rpsc -> crps', I_cr, 
#                     PassengerVehicleFleet_MFA_System.ParameterDict['Powertrain_and_Segment'].Values) 
# I_crp = np.einsum('cr, rpsc -> crp', I_cr, 
#                     PassengerVehicleFleet_MFA_System.ParameterDict['Powertrain_and_Segment'].Values) 
# Ps_rsc =  np.einsum('rpsc -> rsc', PassengerVehicleFleet_MFA_System.ParameterDict['Powertrain_and_Segment'].Values) 
# I_crps =  np.einsum('crp, rsc -> crps', I_crp, Ps_rsc) 
# P_crps = I_crps_corr / I_crps
# # cf.export_to_csv(P_crps, 'P_crps', IndexTable)

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

Al_inflow_cS = np.einsum('crS-> cS ', Al_inflow_crS) 


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
# File flows_scenarios.xlsx, structure taS
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


try:
    df.to_excel('results/flows_scenarios.xlsx', merge_cells=False)
except:
    print('Results could not be saved to results/flows_scenarios.xlsx, the file is probably open')

# File flows_plotly.xlsx, structure tS
F_1_2_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values)
F_2_3_tS = np.einsum('trpszeaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values)
F_3_4_tS = np.einsum('tcrpszeaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values)
F_4_0_tS = np.einsum('tcrpszeaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values)
F_4_5_tS = np.einsum('tcrpszeaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values)
F_4_7_tS = np.einsum('tcrpszeaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values)
F_5_6_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values)
F_5_7_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values)
F_6_0_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values)
F_6_1_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values)
F_7_0_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values)
F_7_1_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values)
F_7_8_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values)
F_8_1_tS = np.einsum('treaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values)
F_0_1_tS = np.einsum('teaS -> tS', PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values)
F_1_9_tS = np.einsum('taS -> tS', scrap_surplus_taS)

iterables = []
names = []
for dim in ['t','S']:
    iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
    names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  

index = pd.MultiIndex.from_product(iterables, names=names)
df = pd.DataFrame(F_2_3_tS.flatten()/10**9,index=index, columns = ['F_2_3'])
df['F_3_4'] = F_3_4_tS.flatten()/10**9
df['F_4_0'] = F_4_0_tS.flatten()/10**9
df['F_4_5'] = F_4_5_tS.flatten()/10**9
df['F_4_7'] = F_4_7_tS.flatten()/10**9
df['F_5_6'] = F_5_6_tS.flatten()/10**9
df['F_5_7'] = F_5_7_tS.flatten()/10**9
df['F_6_0'] = F_6_0_tS.flatten()/10**9
df['F_6_1'] = F_6_1_tS.flatten()/10**9
df['F_7_0'] = F_7_0_tS.flatten()/10**9
df['F_7_1'] = F_7_1_tS.flatten()/10**9
df['F_1_2'] = F_1_2_tS.flatten()/10**9
df['F_1_9'] = F_1_9_tS.flatten()/10**9
df['F_0_1'] = F_0_1_tS.flatten()/10**9
df['F_7_8'] = F_7_8_tS.flatten()/10**9
df['F_8_1'] = F_8_1_tS.flatten()/10**9

try:
    df.to_excel('results/flows_plotly.xlsx', merge_cells=False)
except:
    print('Results could not be saved to results/flows_plotly.xlsx, the file is probably open')

# File flows_per_year.xlsx, structure t
F_0_1_t = np.einsum('taS -> t', F_0_1_taS)/10**9
F_1_2_t = np.einsum('taS -> t', F_1_2_taS)/10**9
F_1_9_t = np.einsum('taS -> t', F_1_9_taS)/10**9
F_2_3_t = np.einsum('taS -> t', F_2_3_taS)/10**9
F_3_4_t = np.einsum('taS -> t', F_3_4_taS)/10**9
F_4_0_t = np.einsum('taS -> t', F_4_0_taS)/10**9
F_4_5_t = np.einsum('taS -> t', F_4_5_taS)/10**9
F_4_7_t = np.einsum('taS -> t', F_4_7_taS)/10**9
F_5_6_t = np.einsum('taS -> t', F_5_6_taS)/10**9
F_5_7_t = np.einsum('taS -> t', F_5_7_taS)/10**9
F_6_0_t = np.einsum('taS -> t', F_6_0_taS)/10**9
F_6_1_t = np.einsum('taS -> t', F_6_1_taS)/10**9
F_7_0_t = np.einsum('taS -> t', F_7_0_taS)/10**9
F_7_1_t = np.einsum('taS -> t', F_7_1_taS)/10**9
F_7_8_t = np.einsum('taS -> t', F_7_8_taS)/10**9
F_8_1_t = np.einsum('taS -> t', F_8_1_taS)/10**9

index = pd.Index(
        PassengerVehicleFleet_MFA_System.IndexTable['Classification']['Time'].Items[:],
        name="Time")

df = pd.DataFrame(F_0_1_t.flatten(),index=index, columns = ['F_0_1'])
df['F_1_2'] = F_1_2_t.flatten()
df['F_1_9'] = F_1_9_t.flatten()
df['F_2_3'] = F_2_3_t.flatten()
df['F_3_4'] = F_3_4_t.flatten()
df['F_4_0'] = F_4_0_t.flatten()
df['F_4_5'] = F_4_5_t.flatten()
df['F_4_7'] = F_4_7_t.flatten()
df['F_5_6'] = F_5_6_t.flatten()
df['F_5_7'] = F_5_7_t.flatten()
df['F_6_0'] = F_6_0_t.flatten()
df['F_6_1'] = F_6_1_t.flatten()
df['F_7_0'] = F_7_0_t.flatten()
df['F_7_1'] = F_7_1_t.flatten()
df['F_7_8'] = F_7_8_t.flatten()
df['F_8_1'] = F_8_1_t.flatten()

try:
    df.to_excel('results/flows_per_year.xlsx')
except:
    print('Results could not be saved to results/flows_per_year.xlsx, the file is probably open')


# %% Plots
start_time = time.time()
print("Plots")
plt.ioff() 
np.seterr(divide='ignore', invalid='ignore') # avoid warning for negative values in divides for first years
fig, ax = plt.subplots()

# all plots are saved in a subfolder named after current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
plot_dir = os.path.join('results', 'plots', current_datetime)

### Scenario comparison plots
## Plot Car Stock per scenario
y_dict = {
        'name': 'Global Car Stock',
        'aspect': 'Scenario',
        'unit': 'Cars'
        }
cf.plot_result_time(S_tS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')

## Plot Al inflows per scenario
y_dict = {
        'name': 'Global Al demand',
        'aspect': 'Scenario',
        'unit': 'Mt/year'
        }
cf.plot_result_time(Al_inflow_cS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')
    
## Plot Total Carbon footprint
y_dict = {
        'name': 'Carbon footprint of Al production',
        'aspect': 'Scenario',
        'unit': 'Mt CO2/yr'
        }
cf.plot_result_time((carbon_footprint_primary + carbon_footprint_secondary)/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')

## Plot Cumulative Carbon footprint
y_dict = {
        'name': 'Cumulative Carbon footprint of Al production',
        'aspect': 'Scenario',
        'unit': 'Mt CO2'
        }
cf.plot_result_time(np.cumsum(carbon_footprint_primary + carbon_footprint_secondary, axis=0)/10**9, y_dict, IndexTable, t_min= 120, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')
        

### Single scenario plots
for scenario in range(NS):

    print("Plotting results for scenario " + str(scenario))
    ## Car Stock per region
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Region',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_trS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows per region
    y_dict = {
            'name': 'Car Inflows',
            'aspect': 'Region',
            'unit': 'cars/yr'
            }
    cf.plot_result_time_scenario(I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    # Car outflows per region
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Region',
            'unit': 'cars/yr'
            }
    cf.plot_result_time_scenario(O_trS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Stock by powertrain
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows by powertrain
    y_dict = {
            'name': 'Car inflows',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Car outflows by powertrain
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Powertrain',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(O_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Stock by segment
    y_dict = {
            'name': 'Car Stock',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(S_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Car inflows by segment
    y_dict = {
            'name': 'Car Inflows',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Car outflows by segment
    y_dict = {
            'name': 'Car Outflows',
            'aspect': 'Segment',
            'unit': 'cars'
            }
    cf.plot_result_time_scenario(O_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
    
    ## Plot Al Stock per component
    y_dict = {
            'name': 'Al Stock',
            'aspect': 'Component',
            'unit': 'Mt'
            }
    cf.plot_result_time_scenario(Al_stock_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Plot Al inflows per conponent
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Component',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_inflow_czS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
    ## Plot Al outflows per conponent
    y_dict = {
            'name': 'Al Outflows',
            
            'aspect': 'Component',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_outflow_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
    
    # Aluminium stock per region
    y_dict = {
            'name': 'Al stock',
            'aspect': 'Region',
            'unit': 'Mt'
            }
    cf.plot_result_time_scenario(Al_stock_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    # Aluminium inflows per region
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Region',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_inflow_crS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    # Aluminium outflows per region
    y_dict = {
            'name': 'Al Outflows',
            'aspect': 'Region',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Al_outflow_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    ## Plot Al Alloys inflows 
    y_dict = {
            'name': 'Al Inflows',
            'aspect': 'Alloy',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Alloys_inflow_caS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    ## Plot Average Al content in inflows per region
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Region',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_crS / I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
    
    ## Plot Average Al content in inflows per powertrain
    I_cpS = np.einsum('crpsS -> cpS', I_crpsS)
    Al_inflow_cpS = np.einsum('crpsS -> cpS', Al_inflow_crpsS)
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Powertrain',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_cpS / I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
    
    ## Plot Average Al content in inflows per segment
    I_csS = np.einsum('crpsS -> csS', I_crpsS)
    Al_inflow_csS = np.einsum('crpsS -> csS', Al_inflow_crpsS)
    y_dict = {
            'name': 'Average Al content',
            'aspect': 'Segment',
            'unit': 'kg/car'
            }
    cf.plot_result_time_scenario(Al_inflow_csS / I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
    ## Plot Al Alloys outflows
    y_dict = {
            'name': 'Al Outflows',
            'aspect': 'Alloy',
            'unit': 'Mt/yr'
            }
    cf.plot_result_time_scenario(Alloys_outflow_taS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    # Plot Ratio Outflows / Inflows
    y_dict = {
            'name': 'Ratio Outflows - Inflows',
            'aspect': 'Alloy',
            'unit': ''
            }
    cf.plot_result_time_scenario(Alloys_outflow_taS / Alloys_inflow_caS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no')
    
    # Plot Ratio Outflows / Inflows for secondary castings per region
    y_dict = {
            'name': 'O-I Ratio for 2nd castings',
            'aspect': 'Region',
            'unit': ''
            }
    cf.plot_result_time_scenario(Alloys_outflow_traS[:,:,2] / Alloys_inflow_craS[:,:,2], y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no')


end_time = time.time()
print("Time for plotting: ", end_time - start_time)


