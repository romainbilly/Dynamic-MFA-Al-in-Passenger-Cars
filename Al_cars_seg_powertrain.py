# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:31:59 2020

Main file for the execution of the code generating the results for the paper
"Aluminium use in passenger cars poses systemic challenges for recycling and GHG emissions"
by Romain G. Billy and Daniel B. Mûller

Warning: export of detailed intermediate results, such as the detailed
composition of the vehicle fleet under different scenarios (S_tcrpsPVLTS.csv)
can use a lot of memory and time, so it is recommended to comment those lines if not needed

@author: romainb
"""

# Load a local copy of the current ODYM branch:
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import logging as log
import time
from datetime import datetime
start_time = time.time()

# add ODYM module directory to system path, relative
MainPath = os.path.join('..', 'odym', 'modules')    
sys.path.insert(0, MainPath)

# add ODYM module directory to system path, absolute
sys.path.insert(0, os.path.join(os.getcwd(), 'odym', 'modules'))

# Specify path to dynamic stock model and to datafile, relative
DataPath = os.path.join('..', 'data')

# Specify path to dynamic stock model and to datafile, absolute
DataPath = os.path.join(os.getcwd(), 'data')

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
Mylog.info('Read model data and parameters.')

# To reload all data from Excel file (if a parameter was changed) uncomment this line:
Data_Prep = DataPrep(DataPath,Mylog)   

# To used previously loaded data from pickle file, use this line (faster):
# Data_Prep = DataPrep(DataPath,Mylog, "ParameterDict.p")

# Default dtype used in numpy arrays
# use 'float64' for extra precision, but default_dtype reduces the memory usage 
# and offers sufficient precision (mass balance is verified at an ~10^-7 precision)
default_dtype = 'float32'


# Mylog.info('### 3 - Read classification and data')


IndexTable = Data_Prep.IndexTable
#Define shortcuts for the most important index sizes:
Nt = len(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)
Nr = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('r')].Items)
Np = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('p')].Items) 
Ns = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('s')].Items) 
Nz = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('z')].Items)
Na = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('a')].Items)
NP = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('P')].Items)
NV = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('V')].Items)
NT = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('T')].Items)
NS = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('S')].Items)
NA = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('A')].Items)
NL = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('L')].Items)
NX = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('X')].Items)
NF = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('F')].Items)
               

Mylog.info('### 5 - Building and solving the dynamic MFA model')
# 1) Determine vehicle inflow and outflow by age-cohort from stock and lifetime data. 
# These calculations are done outside of the MFA system 
# as we are not yet on the material level but at the product level.

O_tcrPVL = np.zeros((Nt,Nt,Nr,NP,NV,NL))
S_tcrPVL= np.zeros((Nt,Nt,Nr,NP,NV,NL))
DS_trPVL = np.zeros((Nt,Nr,NP,NV,NL))
I_crPVL = np.zeros((Nt,Nr,NP,NV,NL))
O_trPVL = np.zeros((Nt,Nr,NP,NV,NL))

S_trPV = np.einsum('Ptr,Vtr -> trPV',
                   Data_Prep.ParameterDict['Population'].Values,
                   Data_Prep.ParameterDict['Vehicle_Ownership'].Values)


Mylog.info('Solving dynamic stock model of the passenger vehicle fleet')
for P in range(NP):
    for V in range(NV):
        for L in range(NL):
            for r in range(Nr):
                # 1a) Loop over all rs to determine a stock-driven model of the global passenger vehicle fleet
                # Create helper DSM for computing the dynamic stock model:
                DSM = dsm.DynamicStockModel(t = np.array(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items),
                                                   s = S_trPV[:,r,P,V], 
                                                   lt = {'Type': 'Normal', 'Mean': Data_Prep.ParameterDict['Vehicle_Lifetime'].Values[L,:,r],
                                                         'StdDev': Data_Prep.ParameterDict['Vehicle_Lifetime'].Values[L,:,r]/4} )
                # lifetime is normally distributed, standard deviation 25% of the mean
                Stock_by_cohort = DSM.compute_stock_driven_model()
            
               
                O_tcrPVL[:,:,r,P,V,L] = DSM.compute_o_c_from_s_c()
                O_trPVL[:,r,P,V,L] = DSM.compute_outflow_total()
                S_tcrPVL[:,:,r,P,V,L] = DSM.s_c
                I_crPVL[:,r,P,V,L] = DSM.i
                DS_trPVL[:,r,P,V,L] = DSM.compute_stock_change()
                
Mylog.info('### 4 - Define MFA system')
Mylog.info('Define MFA system and processes.')

# Shortening to get rid of uneccessary history years
# All years before 2000 and cohorts before 1965 are deleted
Nt = 51
Nc = 86
Data_Prep.IndexTable.Classification[IndexTable.index.get_loc('Time')].Items = \
    Data_Prep.IndexTable.Classification[IndexTable.index.get_loc('Time')].Items[-Nt:] 
Data_Prep.IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('c')].Items = \
    Data_Prep.IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('c')].Items[-Nc:] 
PassengerVehicleFleet_MFA_System = Data_Prep.create_mfa_system(Model_Time_Start = 2051 - Nt)
# Mass balance check:


# Calculating segment split by powertrain 
Mylog.info("Adding Powertrain and Segment splits")
Powertrain_Trpc = PassengerVehicleFleet_MFA_System.ParameterDict['Powertrains'].Values
Segment_Srsc = PassengerVehicleFleet_MFA_System.ParameterDict['Segments'].Values
PS_TSrpsc =  np.einsum('Trpc, Srsc -> TSrpsc', Powertrain_Trpc, Segment_Srsc)

# Correction according to SP_Coeff parameter
PS_TSrpsc =  np.einsum('TSrpsc, psc -> TSrpsc', PS_TSrpsc,
                      PassengerVehicleFleet_MFA_System.ParameterDict['SP_Coeff'].Values)

for S in range(NS):
    for T in range(NT):
        PS_TSrpsc[T,S,:,:,:,:] =  np.einsum('rpsc, rpc -> rpsc', PS_TSrpsc[T,S,:,:,:,:], np.nan_to_num(Powertrain_Trpc[T,:,:,:] / (np.sum(PS_TSrpsc[T,S,:,:,:,:], axis=2))))
        PS_TSrpsc[T,S,:,:,:,:] =  np.einsum('rpsc, rsc -> rpsc', PS_TSrpsc[T,S,:,:,:,:], np.nan_to_num(Segment_Srsc[S,:,:,:] / (np.sum(PS_TSrpsc[T,S,:,:,:,:], axis=1))))
# for powertrains HEV, PHEV, and BEV, a correction coefficient is applied to the segments AB, DE, and SUV. 
# Segment C is calculated by "mass balance" to reach the average powertrain split 
for S in range(NS):
    for T in range(NT):
        PS_TSrpsc[T,S,:,1,1,:] = Powertrain_Trpc[T,:,1,:] - PS_TSrpsc[T,S,:,1,0,:] - PS_TSrpsc[T,S,:,1,2,:] - PS_TSrpsc[T,S,:,1,3,:]
        PS_TSrpsc[T,S,:,2,1,:] = Powertrain_Trpc[T,:,2,:] - PS_TSrpsc[T,S,:,2,0,:] - PS_TSrpsc[T,S,:,2,2,:] - PS_TSrpsc[T,S,:,2,3,:]
        PS_TSrpsc[T,S,:,3,1,:] = Powertrain_Trpc[T,:,3,:] - PS_TSrpsc[T,S,:,3,0,:] - PS_TSrpsc[T,S,:,3,2,:] - PS_TSrpsc[T,S,:,3,3,:]
    # the segment split of the ICEV segment is calculated from the other powertrain types to reach the average segment split
        for s in range(Ns):
            PS_TSrpsc[T,S,:,0,s,:] = Segment_Srsc[S,:,s,:] -  np.sum(PS_TSrpsc[T,S,:,1:,s,:], axis=1)

PS_TSrpsc[np.isclose(PS_TSrpsc, 0, atol=10**(-6))] = 0
negative_values = np.sum((np.array(PS_TSrpsc < 0)))
if negative_values > 0: 
    Mylog.info("WARNING: ", negative_values, " negative values are present in the type/segment matrix, please check the data")
    # export the detailed PS_TSrpsc matric to check for negative values
    cf.export_to_csv(PS_TSrpsc[:,:,:,:,:,-Nc:], 'PS_TSrpsc', IndexTable)

# Stock py powertrain and segment with scenarios
S_tcrpsPVLTS = np.einsum('tcrPVL, TSrpsc -> tcrpsPVLTS', S_tcrPVL, PS_TSrpsc).astype(default_dtype)
I_crpsPVLTS = np.einsum('crPVL, TSrpsc -> crpsPVLTS', I_crPVL, PS_TSrpsc).astype(default_dtype)
O_tcrpsPVLTS = np.einsum('tcrPVL, TSrpsc -> tcrpsPVLTS', O_tcrPVL, PS_TSrpsc).astype(default_dtype)



S_tcrpsPVLTS_short = S_tcrpsPVLTS[-Nt:,-Nc:,:,:,:,:,:,:,:].astype(default_dtype)
I_crpsPVLTS_short = I_crpsPVLTS[-Nc:,:,:,:,:,:,:,:].astype(default_dtype)
O_tcrpsPVLTS_short = O_tcrpsPVLTS[-Nt:,-Nc:,:,:,:,:,:,:,:].astype(default_dtype)


#Aluminium content calculations by scenario, corrected by P_seg and P_type
Mylog.info("Performing Al content calculations")

# Stock
Al_stock_tcrpsPVLTSA = np.einsum('tcrpsPVLTS, Arc, sc, pc -> tcrpsPVLTSA', S_tcrpsPVLTS_short, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values[:,:,-Nc:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values[:,-Nc:],                  
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values[:,-Nc:], optimize=True).astype(default_dtype)
# Inflow
Al_inflow_crpsPVLTSA = np.einsum('crpsPVLTS, Arc, sc, pc -> crpsPVLTSA', I_crpsPVLTS_short, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values[:,:,-Nc:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values[:,-Nc:],                  
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values[:,-Nc:], optimize=True).astype(default_dtype)
# Outflow
Al_outflow_tcrpsPVLTSA = np.einsum('tcrpsPVLTS, Arc, sc, pc -> tcrpsPVLTSA', O_tcrpsPVLTS_short, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Aluminium_Content'].Values[:,:,-Nc:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_seg'].Values[:,-Nc:],                  
                   PassengerVehicleFleet_MFA_System.ParameterDict['P_type'].Values[:,-Nc:], optimize=True).astype(default_dtype)
end_time = time.time()
Mylog.info(end_time-start_time)


# Component level calculations
start_time = time.time()
Mylog.info("Performing component level and alloy content calculations")

# Stock
Al_stock_tcrpsaPVLTSA = np.einsum('tcrpsPVLTSA, crpsz, az -> tcrpsaPVLTSA', Al_stock_tcrpsPVLTSA, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values[-Nc:,:,:,:,:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values, optimize=True).astype(default_dtype) 
# Inflow
Al_inflow_crpszaPVLTSA = np.einsum('crpsPVLTSA, crpsz, az -> crpszaPVLTSA', Al_inflow_crpsPVLTSA, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values[-Nc:,:,:,:,:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values, optimize=True).astype(default_dtype)  
# Outflow
Al_outflow_trpszaPVLTSA = np.einsum('tcrpsPVLTSA, crpsz, az -> trpszaPVLTSA', Al_outflow_tcrpsPVLTSA, 
                   PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values[-Nc:,:,:,:,:],
                   PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values, optimize=True).astype(default_dtype) 
end_time = time.time()
Mylog.info(end_time-start_time)


start_time = time.time()
# Solving the MFA system
Mylog.info("Solving the MFA system")

# S_3, dimensions 't,c,r,p,s,a,P,V,T,S,A'
PassengerVehicleFleet_MFA_System.StockDict['S_3'].Values = Al_stock_tcrpsaPVLTSA
# F_2_3, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values = \
        np.einsum('crpszaPVLTSA -> crpsaPVLTSA', Al_inflow_crpszaPVLTSA[-Nt:,:,:,:,:,:,:,:,:,:,:].astype(default_dtype)).astype(default_dtype)
# F_1_2, Materials for Passenger vehicle production, dimensions t,r,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values = \
        np.einsum('crpsaPVLTSA-> craPVLTSA',  PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values.astype(default_dtype)).astype(default_dtype)
# F_3_4, EoL Vehicles, dimensions 't,r,p,s,z,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values = Al_outflow_trpszaPVLTSA[-Nt:,:,:,:,:,:,:,:,:,:,:].astype(default_dtype)
# F_4_0, dimensions 't,r,p,s,z,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values = \
    np.einsum('trpszaPVLTSA, tr -> trpszaPVLTSA',
    PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype),
    1 - PassengerVehicleFleet_MFA_System.ParameterDict['Collection'].Values[-Nt:,:].astype(default_dtype)).astype(default_dtype)

# F_4_5, Collected cars to dismantling, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values = \
    np.einsum('trpszaPVLTSA, rzt -> trpsaPVLTSA',
    PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype) - \
    PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values.astype(default_dtype),
    PassengerVehicleFleet_MFA_System.ParameterDict['Dismantling'].Values[:,:,-Nt:].astype(default_dtype)).astype(default_dtype)
# F_4_7, Collected cars to shredding, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values = \
        np.einsum('trpszaPVLTSA -> trpsaPVLTSA',
                  PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype) - \
                  PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values.astype(default_dtype)).astype(default_dtype) - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values.astype(default_dtype)       
# F_5_6, Dismantled components to shredding, dimensions 't,r,a,P,V,L,T,S,A'
dismantling_yield = 0.7
PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values = \
        dismantling_yield * np.einsum('trpsaPVLTSA-> traPVLTSA',
                        PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values).astype(default_dtype)                                         
# F_5_7, Residues from dismantllng to shredding, dimensions t,r,a,P,V,L,T,S,A
# need to add dismantling yield
PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values = \
       (1 - dismantling_yield) * \
       np.einsum('trpsaPVLTSA -> traPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values).astype(default_dtype) 
# F_6_1, Al scrap from dismantled components, dimensions t,r,a,P,V,L,T,S,A
# Definition of shredding and sorting yield
recovery_yield = 0.85
PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values = \
        recovery_yield * PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values.astype(default_dtype) 
# F_6_0, Shredding losses, dimensions t,r,a,P,V,L,T,S,A
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values = \
        (1 - recovery_yield) * PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values.astype(default_dtype)
# F_7_0, Shredding losses, dimensions t,r,a,P,V,L,T,S,A
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values =  (1 - recovery_yield) * (
        np.einsum('trpsaPVLTSA-> traPVLTSA', 
                  PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values).astype(default_dtype)
# F_7_8, Scrap to alloy sorting, dimensions t,r,a,P,V,L,T,S,A,X
PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values = np.empty((Nt,Nr,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)
            
for X in range(NX):
    PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values[:,:,:,:,:,:,:,:,:,X] = \
    np.einsum('traPVLTSA, tr -> traPVLTSA',
            np.einsum('trpsaPVLTSA-> traPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values,
            PassengerVehicleFleet_MFA_System.ParameterDict['Alloy_Sorting'].Values[X,-Nt:,:],
            optimize=True)
# F_7_1, Mixed shredded scrap, dimensions t,r,a,P,V,L,T,S,A,X
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values =np.empty((Nt,Nr,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)
for X in range(NX):
    PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[...,X] = \
            np.einsum('trpsaPVLTSA-> traPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values[...,X]
# Alloy composition adjusted to become secondary castings / mixed scrap only
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[:,:,2,...] =\
        np.einsum('traPVLTSAX -> trPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values)  
# Setting wrought and primary casting to zero        
PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values = \
        np.einsum('traPVLTSAX, a -> traPVLTSAX', 
                  PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values, 
                  np.array([0,0,1]))
# F_8_1, Alloy sorted scrap, dimensions t,r,a,P,V,L,T,S,A,X
PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values = PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values

    
# Correcting for scrap surplus
# Scrap surplus considered at global level only
Mylog.info("Correcting for scrap surplus")
# Mass balance of process 1 without scrap surplus and primary production
# If positive, there is a scrap surplus for the alloy considered
Process_1_mb_taPVLTSAX = np.zeros((Nt,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)
scrap_surplus_taPVLTSAX = np.zeros((Nt,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)      

for X in range(NX):
    Process_1_mb_taPVLTSAX[...,X] = np.einsum('traPVLTSA-> taPVLTSA', 
        PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values.astype(default_dtype) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[...,X].astype(default_dtype) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values[...,X].astype(default_dtype) - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values).astype(default_dtype)


for it,ia,iP,iV,iL,iT,iS,iA,iX in np.ndindex(Process_1_mb_taPVLTSAX.shape):
    if Process_1_mb_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX] > 0:
        scrap_surplus_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX] = Process_1_mb_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX].astype(default_dtype)

PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values = scrap_surplus_taPVLTSAX.astype(default_dtype)  

# F_0_1, Primary Aluminium Demand, determined by mass balance
for X in range(NX):
    PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values[...,X] = \
            np.einsum('traPVLTSA-> taPVLTSA', 
            PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[...,X] - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values[...,X]) + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values[...,X]     
        

# dS_3, dimensions 't,r,p,s,a,P,V,T,S,A' 
PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values = \
    PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values - \
    np.einsum('trpszaPVLTSA-> trpsaPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values)
                                                            
#### Carbon footprint calculations   
carbon_footprint_primary = np.einsum('taPVLTSAX, tF -> tPVLTSAXF', 
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values,
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Primary'].Values[-Nt:,:]).astype(default_dtype)

carbon_footprint_secondary = np.zeros(carbon_footprint_primary.shape, default_dtype)
for X in range(NX):
    carbon_footprint_secondary[...,X,:] = np.einsum('taPVLTSA, tF -> tPVLTSAF', 
                                     np.einsum('traPVLTSA -> taPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values) -\
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values[...,X],
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Secondary'].Values[-Nt:,:])
     
end_time = time.time()
Mylog.info(end_time-start_time)   

# Exports
Mylog.info("Exporting data")
# Raw files for detailed vehicle fleet composition

# Detailed inflow composition
Mylog.info("Exporting to I_crpsPVLTS.csv")
cf.export_to_csv(I_crpsPVLTS_short, 'I_crpsPVLTS', IndexTable)

# Detailed stock composition
Mylog.info("Exporting to S_tcrpsPVLTS.csv")
#cf.export_to_csv(S_tcrpsPVLTS_short, 'S_tcrpsPVLTS', IndexTable)


# File flows_scenarios_parameters.csv, structure taPVLTSA
Mylog.info("Exporting to flows_scenarios_parameters.csv")
X = np.ones(NX)
F_1_2_taPVLTSAX = np.einsum('traPVLTSA, X -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values, X)        
F_2_3_taPVLTSAX = np.einsum('trpsaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values, X)
F_3_4_taPVLTSAX = np.einsum('trpszaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values, X)
F_4_0_taPVLTSAX = np.einsum('trpszaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values, X)
F_4_5_taPVLTSAX = np.einsum('trpsaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values, X)
F_4_7_taPVLTSAX = np.einsum('trpsaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values, X)
F_5_6_taPVLTSAX = np.einsum('traPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values, X)
F_5_7_taPVLTSAX = np.einsum('traPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values, X)
F_6_0_taPVLTSAX = np.einsum('traPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values, X)
F_6_1_taPVLTSAX = np.einsum('traPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values, X)
F_7_0_taPVLTSAX = np.einsum('traPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values, X)
F_7_1_taPVLTSAX = np.einsum('traPVLTSAX -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values)
F_7_8_taPVLTSAX = np.einsum('traPVLTSAX -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values)
F_8_1_taPVLTSAX = np.einsum('traPVLTSAX -> taPVLTSAX', PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values)
F_0_1_taPVLTSAX = PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values
F_1_9_taPVLTSAX = scrap_surplus_taPVLTSAX
S_3_taPVLTSAX = np.einsum('tcrpsaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.StockDict['S_3'].Values, X)
dS_3_taPVLTSAX = np.einsum('trpsaPVLTSA,X  -> taPVLTSAX', PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values, X)

iterables = []
names = []
for dim in ['t','a','P','V','L','T','S','A','X']:
    iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
    names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  

index = pd.MultiIndex.from_product(iterables, names=names)
df = pd.DataFrame(F_2_3_taPVLTSAX.flatten()/10**9,index=index, columns = ['F_2_3_ta'])
df['F_3_4_ta'] = F_3_4_taPVLTSAX.flatten()/10**9
df['F_4_0_ta'] = F_4_0_taPVLTSAX.flatten()/10**9
df['F_4_5_ta'] = F_4_5_taPVLTSAX.flatten()/10**9
df['F_4_7_ta'] = F_4_7_taPVLTSAX.flatten()/10**9
df['F_5_6_ta'] = F_5_6_taPVLTSAX.flatten()/10**9
df['F_5_7_ta'] = F_5_7_taPVLTSAX.flatten()/10**9
df['F_6_0_ta'] = F_6_0_taPVLTSAX.flatten()/10**9
df['F_6_1_ta'] = F_6_1_taPVLTSAX.flatten()/10**9
df['F_7_0_ta'] = F_7_0_taPVLTSAX.flatten()/10**9
df['F_7_1_ta'] = F_7_1_taPVLTSAX.flatten()/10**9
df['F_1_2_ta'] = F_1_2_taPVLTSAX.flatten()/10**9
df['F_1_9_ta'] = scrap_surplus_taPVLTSAX.flatten()/10**9
df['S_3_ta'] = S_3_taPVLTSAX.flatten()/10**9
df['dS_3_ta'] = dS_3_taPVLTSAX.flatten()/10**9
df['F_0_1_ta'] = F_0_1_taPVLTSAX.flatten()/10**9
df['F_7_8_ta'] = F_7_8_taPVLTSAX.flatten()/10**9
df['F_8_1_ta'] = F_8_1_taPVLTSAX.flatten()/10**9

path = 'results/flows_scenarios_parameters.csv' 
cf.export_df_to_csv(df, path)
df.to_pickle('results/flows_scenarios_parameters.pkl')

Mylog.info('checking mass balance of process 7')
balance = np.einsum('taPVLTSAX -> tPVLTSAX', F_4_7_taPVLTSAX) + \
            np.einsum('taPVLTSAX -> tPVLTSAX', F_5_7_taPVLTSAX) - \
            np.einsum('taPVLTSAX -> tPVLTSAX', F_7_0_taPVLTSAX) - \
            np.einsum('taPVLTSAX -> tPVLTSAX', F_7_1_taPVLTSAX) - \
            np.einsum('taPVLTSAX -> tPVLTSAX', F_7_8_taPVLTSAX) 
sum_inflows = np.einsum('taPVLTSAX -> tPVLTSAX',  F_4_7_taPVLTSAX + F_5_7_taPVLTSAX)
Mylog.info('Maximum relative MBI process 7')
Mylog.info(np.max(balance/sum_inflows))
                
Mylog.info('checking mass balance of process 1')
balance = F_0_1_taPVLTSAX + F_6_1_taPVLTSAX + F_7_1_taPVLTSAX + F_8_1_taPVLTSAX - \
    F_1_2_taPVLTSAX - F_1_9_taPVLTSAX
sum_inflows = F_0_1_taPVLTSAX + F_6_1_taPVLTSAX + F_7_1_taPVLTSAX + F_8_1_taPVLTSAX
Mylog.info('Maximum relative MBI process 1')
Mylog.info(np.max(balance/sum_inflows))

# File Carbon_footprint_scenarios_parameters.xlsx, structure tPVTSA
iterables = []
names = []
for dim in ['t','P','V','L','T','S','A','X','F']:
    iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
    names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0])  

index = pd.MultiIndex.from_product(iterables, names=names)
df = pd.DataFrame(carbon_footprint_primary.flatten()/10**9,index=index, columns = ['Carbon_footprint_primary'])
df['Carbon_footprint_secondary'] = carbon_footprint_secondary.flatten()/10**9
df['Carbon_footprint_total'] = (carbon_footprint_primary + carbon_footprint_secondary).flatten()/10**9

path = 'results/carbon_footprint_scenarios_parameters.csv' 
cf.export_df_to_csv(df, path)
 

# File flows_plotly_scenarios_parameters.csv, structure tPVLTSAX
# export file used to generate the interactive Sankey visualisation
iterables = []
names = []
for dim in ['t','P','V','L','T','S','A','X']:
    iterables.append(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(dim)].Items)
    names.append(IndexTable[IndexTable['IndexLetter'] == dim]['Description'].index.values[0]) 

index = pd.MultiIndex.from_product(iterables, names=names)
df = pd.DataFrame(np.einsum('taPVLTSAX -> tPVLTSAX', F_1_2_taPVLTSAX).flatten()/10**9 ,
                  index=index, columns = ['F_1_2'])
df['F_2_3'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_2_3_taPVLTSAX).flatten()/10**9
df['F_3_4'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_3_4_taPVLTSAX).flatten()/10**9
df['F_4_0'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_4_0_taPVLTSAX).flatten()/10**9
df['F_4_5'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_4_5_taPVLTSAX).flatten()/10**9
df['F_4_7'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_4_7_taPVLTSAX).flatten()/10**9
df['F_5_6'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_5_6_taPVLTSAX).flatten()/10**9
df['F_5_7'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_5_7_taPVLTSAX).flatten()/10**9
df['F_6_0'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_6_0_taPVLTSAX).flatten()/10**9
df['F_6_1'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_6_1_taPVLTSAX).flatten()/10**9
df['F_7_0'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_7_0_taPVLTSAX).flatten()/10**9
df['F_7_1'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_7_1_taPVLTSAX).flatten()/10**9
df['F_7_8'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_7_8_taPVLTSAX).flatten()/10**9
df['F_8_1'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_8_1_taPVLTSAX).flatten()/10**9
df['F_0_1'] = np.einsum('taPVLTSAX -> tPVLTSAX', F_0_1_taPVLTSAX).flatten()/10**9
df['F_1_9'] = np.einsum('taPVLTSAX -> tPVLTSAX', scrap_surplus_taPVLTSAX).flatten()/10**9

balance = df['F_5_7'] + df['F_4_7'] - df['F_7_0'] - df['F_7_0'] - df['F_7_8']
Mylog.info(np.max(balance))


path = 'results/flows_plotly_parameters.csv' 
cf.export_df_to_csv(df, path)





