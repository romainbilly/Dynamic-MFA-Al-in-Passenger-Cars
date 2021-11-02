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

# Data_Prep = DataPrep(DataPath,Mylog)      
Data_Prep = DataPrep(DataPath,Mylog, "ParameterDict.p")

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
# for letter in IndexTable.IndexLetter:
#     vars()['N' + str(letter)] = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc(letter)].Items)



                      

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
# Nc = 86
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

# cf.export_to_csv(PS_TSrpsc, 'PS_TSrpsc', IndexTable)

# Stock py powertrain and segment with scenarios
S_tcrpsPVLTS = np.einsum('tcrPVL, TSrpsc -> tcrpsPVLTS', S_tcrPVL, PS_TSrpsc).astype(default_dtype)
I_crpsPVLTS = np.einsum('crPVL, TSrpsc -> crpsPVLTS', I_crPVL, PS_TSrpsc).astype(default_dtype)
O_tcrpsPVLTS = np.einsum('tcrPVL, TSrpsc -> tcrpsPVLTS', O_tcrPVL, PS_TSrpsc).astype(default_dtype)



S_tcrpsPVLTS_short = S_tcrpsPVLTS[-Nt:,-Nc:,:,:,:,:,:,:,:].astype(default_dtype)
I_crpsPVLTS_short = I_crpsPVLTS[-Nc:,:,:,:,:,:,:,:].astype(default_dtype)
O_tcrpsPVLTS_short = O_tcrpsPVLTS[-Nt:,-Nc:,:,:,:,:,:,:,:].astype(default_dtype)

# for T in range(NT):
#     for S in range(NS):
#         Mylog.info(T,S,np.sum(np.array(PS_TSrpsc[T,S,:,:,:,:] < 0)))

# for p in range(Np):
#     for s in range(Ns):
#         Mylog.info(p,s,np.sum(np.array(PS_TSrpsc[:,:,:,p,s,:] < 0)))

# for c in range(Nt):
#     Mylog.info(c,np.sum(np.array(PS_TSrpsc[:,:,:,:,:,c] < 0)))


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
# Al_stock_trpszaPVLTSA = np.einsum('tcrpsPVLTSA, crpsz, az -> trpszaPVLTSA', Al_stock_tcrpsPVLTSA, 
#                    PassengerVehicleFleet_MFA_System.ParameterDict['Components'].Values[-Nc:,:,:,:,:],
#                    PassengerVehicleFleet_MFA_System.ParameterDict['Alloys'].Values, optimize=True) 
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
print(PassengerVehicleFleet_MFA_System.StockDict['S_3'].Values.dtype)
# F_2_3, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values = \
        np.einsum('crpszaPVLTSA -> crpsaPVLTSA', Al_inflow_crpszaPVLTSA[-Nt:,:,:,:,:,:,:,:,:,:,:].astype(default_dtype)).astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values.dtype)
# F_1_2, Materials for Passenger vehicle production, dimensions t,r,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values = \
        np.einsum('crpsaPVLTSA-> craPVLTSA',  PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values.astype(default_dtype)).astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values.dtype)
# F_3_4, EoL Vehicles, dimensions 't,r,p,s,z,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values = Al_outflow_trpszaPVLTSA[-Nt:,:,:,:,:,:,:,:,:,:,:].astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.dtype)
# F_4_0, dimensions 't,r,p,s,z,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values = \
    np.einsum('trpszaPVLTSA, tr -> trpszaPVLTSA',
    PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype),
    1 - PassengerVehicleFleet_MFA_System.ParameterDict['Collection'].Values[-Nt:,:].astype(default_dtype)).astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values.dtype)

# F_4_5, Collected cars to dismantling, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values = \
    np.einsum('trpszaPVLTSA, rzt -> trpsaPVLTSA',
    PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype) - \
    PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values.astype(default_dtype),
    PassengerVehicleFleet_MFA_System.ParameterDict['Dismantling'].Values[:,:,-Nt:].astype(default_dtype)).astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values.dtype)
# F_4_7, Collected cars to shredding, dimensions 't,r,p,s,a,P,V,L,T,S,A'
PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values = \
        np.einsum('trpszaPVLTSA -> trpsaPVLTSA',
                  PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values.astype(default_dtype) - \
                  PassengerVehicleFleet_MFA_System.FlowDict['F_4_0'].Values.astype(default_dtype)).astype(default_dtype) - \
        PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values.astype(default_dtype)       
print(PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values.dtype)
# F_5_6, Dismantled components to shredding, dimensions 't,r,a,P,V,L,T,S,A'
# need to add dismantling yield
dismantling_yield = 0.7
PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values = \
        dismantling_yield * np.einsum('trpsaPVLTSA-> traPVLTSA',
                        PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values).astype(default_dtype)                                         
print(PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values.dtype)
# F_5_7, Residues from dismantllng to shredding, dimensions t,r,a,P,V,L,T,S,A
# need to add dismantling yield
PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values = \
       (1 - dismantling_yield) * \
       np.einsum('trpsaPVLTSA -> traPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_4_5'].Values).astype(default_dtype) 
print(PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values.dtype)
# F_6_1, Al scrap from dismantled components, dimensions t,r,a,P,V,L,T,S,A
# need to add shredding yield
shredding_yield = 0.95
PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values = \
        shredding_yield * PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values.astype(default_dtype) 
print(PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values.dtype)
# F_6_0, Shredding losses, dimensions t,r,a,P,V,L,T,S,A
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values = \
        (1 - shredding_yield) * PassengerVehicleFleet_MFA_System.FlowDict['F_5_6'].Values.astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_6_0'].Values.dtype)
# F_7_0, Shredding losses, dimensions t,r,a,P,V,L,T,S,A
# need to add shredding yield
PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values =  (1 - shredding_yield) * (
        np.einsum('trpsaPVLTSA-> traPVLTSA', 
                  PassengerVehicleFleet_MFA_System.FlowDict['F_4_7'].Values) + \
        PassengerVehicleFleet_MFA_System.FlowDict['F_5_7'].Values).astype(default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_7_0'].Values.dtype)
# F_7_8, Scrap to alloy sorting, dimensions t,r,a,P,V,L,T,S,A,X
PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values = np.empty((Nt,Nr,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)
print(PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values.dtype)
            
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
print(PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values.dtype)
# F_8_1, Alloy sorted scrap, dimensions t,r,a,P,V,L,T,S,A,X
PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values = PassengerVehicleFleet_MFA_System.FlowDict['F_7_8'].Values
print(PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values.dtype)

    
# Correcting for scrap surplus
# Scrap surplus considered at global level only
Mylog.info("Correcting for scrap surplus")
# Mass balance of process 1 without scrap surplus and primary production
# If positive, there is a scrap surplus for the alloy considered
Process_1_mb_taPVLTSAX = np.zeros((Nt,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)
scrap_surplus_taPVLTSAX = np.zeros((Nt,Na,NP,NV,NL,NT,NS,NA,NX),default_dtype)      

for X in range(NX):
    Process_1_mb_taPVLTSAX[...,X] = np.einsum('traPVLTSA-> taPVLTSA', 
            PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values.astype(default_dtype + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[...,X].astype(default_dtype + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values[...,X].astype(default_dtype - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values).astype(default_dtype)


for it,ia,iP,iV,iL,iT,iS,iA,iX in np.ndindex(Process_1_mb_taPVLTSAX.shape):
    if Process_1_mb_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX] > 0:
        scrap_surplus_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX] = Process_1_mb_taPVLTSAX[it,ia,iP,iV,iL,iT,iS,iA,iX].astype(default_dtype)

PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values = scrap_surplus_taPVLTSAX.astype(default_dtype)  
print(PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values.dtype)

# F_0_1, Primary Aluminium Demand, determined by mass balance
for X in range(NX):
    PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values[...,X] = \
            np.einsum('traPVLTSA-> taPVLTSA', 
            PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_6_1'].Values - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_7_1'].Values[...,X] - \
            PassengerVehicleFleet_MFA_System.FlowDict['F_8_1'].Values[...,X]) + \
            PassengerVehicleFleet_MFA_System.FlowDict['F_1_9'].Values[...,X]     
print(PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values.dtype)
        

# dS_3, dimensions 't,r,p,s,a,P,V,T,S,A' 
PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values = \
    PassengerVehicleFleet_MFA_System.FlowDict['F_2_3'].Values - \
    np.einsum('trpszaPVLTSA-> trpsaPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_3_4'].Values)
print(PassengerVehicleFleet_MFA_System.StockDict['dS_3'].Values.dtype)
                                                            
#### Carbon footprint calculations   
carbon_footprint_primary = np.einsum('taPVLTSAX, tF -> tPVLTSAXF', 
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values,
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Primary'].Values[-Nt:,:]).astype(default_dtype)
print(carbon_footprint_primary.dtype)

carbon_footprint_secondary = np.zeros(carbon_footprint_primary.shape, default_dtype)
for X in range(NX):
    carbon_footprint_secondary[...,X,:] = np.einsum('taPVLTSA, tF -> tPVLTSAF', 
                                     np.einsum('traPVLTSA -> taPVLTSA', PassengerVehicleFleet_MFA_System.FlowDict['F_1_2'].Values) -\
                                     PassengerVehicleFleet_MFA_System.FlowDict['F_0_1'].Values[...,X],
                                     PassengerVehicleFleet_MFA_System.ParameterDict['Carbon_Footprint_Secondary'].Values[-Nt:,:])
print(carbon_footprint_secondary.dtype)   
                                               
# Mass balance check:
# Mylog.info("Checking Mass Balance")    
# Bal = PassengerVehicleFleet_MFA_System.MassBalanceNoElement()
# Mylog.info(np.abs(Bal).sum(axis = 0)) # reports the sum of all absolute balancing errors by process.        

        # Bal = np.zeros((len(self.Time_L),len(self.ProcessList))) # Balance array: years x process x element: 
        # #process position 0 is the balance for the system boundary, the other positions are for the processes, 
        # #element position 0 is the balance for the entire mass, the other are for the balance of the individual elements
        
        # for key in self.FlowDict: # Add all flows to mass balance
        #     Bal[:,self.FlowDict[key].P_Start] -= self.Flow_Sum(key) # Flow leaving a process
        #     Bal[:,self.FlowDict[key].P_End]   += self.Flow_Sum(key) # Flow entering a process
            
        # for key in self.StockDict: # Add all stock changes to the mass balance
        #     if  self.StockDict[key].Type == 1:
        #         Bal[:,self.StockDict[key].P_Res] -= self.Stock_Sum(key) # 1: net stock change or addition to stock
        #     elif self.StockDict[key].Type == 2:
        #         Bal[:,self.StockDict[key].P_Res] += self.Stock_Sum(key) # 2: removal/release from stock
            
        # #add stock changes to process with number 0 ('system boundary, environment of system')
        # for key in self.StockDict:
        #     if  self.StockDict[key].Type == 1:
        #         Bal[:,0] += self.Stock_Sum(key) # 1: net stock change or addition to stock
        #     elif self.StockDict[key].Type == 2:
        #         Bal[:,0] -= self.Stock_Sum(key) # 2: removal/release from stock
            
        # return Bal


end_time = time.time()
Mylog.info(end_time-start_time)   

# Exports
Mylog.info("Exporting data")
# Raw files vehicle fleet
# cf.export_to_csv(I_crpsPVLTS_short, 'I_crpsPVLTS', IndexTable)
cf.export_to_csv(S_tcrpsPVLTS_short, 'S_tcrpsPVLTS', IndexTable)
# cf.export_to_csv(O_tcrpsS, 'O_tcrpsS', IndexTable)

# File flows_scenarios_parameters.csv, structure taPVLTSA
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
 
# File flows_plotly_scenarios_parameters.xlsx, structure tPVLTSAX
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











# # File flows_per_year.xlsx, structure t
# F_0_1_t = np.einsum('taS -> t', F_0_1_taS)/10**9
# F_1_2_t = np.einsum('taS -> t', F_1_2_taS)/10**9
# F_1_9_t = np.einsum('taS -> t', F_1_9_taS)/10**9
# F_2_3_t = np.einsum('taS -> t', F_2_3_taS)/10**9
# F_3_4_t = np.einsum('taS -> t', F_3_4_taS)/10**9
# F_4_0_t = np.einsum('taS -> t', F_4_0_taS)/10**9
# F_4_5_t = np.einsum('taS -> t', F_4_5_taS)/10**9
# F_4_7_t = np.einsum('taS -> t', F_4_7_taS)/10**9
# F_5_6_t = np.einsum('taS -> t', F_5_6_taS)/10**9
# F_5_7_t = np.einsum('taS -> t', F_5_7_taS)/10**9
# F_6_0_t = np.einsum('taS -> t', F_6_0_taS)/10**9
# F_6_1_t = np.einsum('taS -> t', F_6_1_taS)/10**9
# F_7_0_t = np.einsum('taS -> t', F_7_0_taS)/10**9
# F_7_1_t = np.einsum('taS -> t', F_7_1_taS)/10**9
# F_7_8_t = np.einsum('taS -> t', F_7_8_taS)/10**9
# F_8_1_t = np.einsum('taS -> t', F_8_1_taS)/10**9

# index = pd.Index(
#         PassengerVehicleFleet_MFA_System.IndexTable['Classification']['Time'].Items[:],
#         name="Time")

# df = pd.DataFrame(F_0_1_t.flatten(),index=index, columns = ['F_0_1'])
# df['F_1_2'] = F_1_2_t.flatten()
# df['F_1_9'] = F_1_9_t.flatten()
# df['F_2_3'] = F_2_3_t.flatten()
# df['F_3_4'] = F_3_4_t.flatten()
# df['F_4_0'] = F_4_0_t.flatten()
# df['F_4_5'] = F_4_5_t.flatten()
# df['F_4_7'] = F_4_7_t.flatten()
# df['F_5_6'] = F_5_6_t.flatten()
# df['F_5_7'] = F_5_7_t.flatten()
# df['F_6_0'] = F_6_0_t.flatten()
# df['F_6_1'] = F_6_1_t.flatten()
# df['F_7_0'] = F_7_0_t.flatten()
# df['F_7_1'] = F_7_1_t.flatten()
# df['F_7_8'] = F_7_8_t.flatten()
# df['F_8_1'] = F_8_1_t.flatten()

# try:
#     df.to_excel('results/flows_per_year.xlsx')
# except:
#     Mylog.info('Results could not be saved to results/flows_per_year.xlsx, the file is probably open')
    
    



# # %% Plots
# start_time = time.time()
# Mylog.info("Plots")
# plt.ioff() 
# np.seterr(divide='ignore', invalid='ignore') # avoid warning for negative values in divides for first years
# fig, ax = plt.subplots()

# # all plots are saved in a subfolder named after current date and time
# current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
# plot_dir = os.path.join('results', 'plots', current_datetime)

# ### Scenario comparison plots
# ## Plot Car Stock per scenario
# y_dict = {
#         'name': 'Global Car Stock',
#         'aspect': 'Scenario',
#         'unit': 'Cars'
#         }
# cf.plot_result_time(S_tS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')

# ## Plot Al inflows per scenario
# y_dict = {
#         'name': 'Global Al demand',
#         'aspect': 'Scenario',
#         'unit': 'Mt/year'
#         }
# cf.plot_result_time(Al_inflow_cS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')
    
# ## Plot Total Carbon footprint
# y_dict = {
#         'name': 'Carbon footprint of Al production',
#         'aspect': 'Scenario',
#         'unit': 'Mt CO2/yr'
#         }
# cf.plot_result_time((carbon_footprint_primary + carbon_footprint_secondary)/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')

# ## Plot Cumulative Carbon footprint
# y_dict = {
#         'name': 'Cumulative Carbon footprint of Al production',
#         'aspect': 'Scenario',
#         'unit': 'Gt CO2'
#         }
# cf.plot_result_time(np.cumsum(carbon_footprint_primary + carbon_footprint_secondary, axis=0)/10**12, y_dict, IndexTable, t_min= 120, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')
        
# ## Plot Scrap surplus
# y_dict = {
#         'name': 'Global scrap surplus',
#         'aspect': 'Scenario',
#         'unit': 'Mt Al'
#         }
# cf.plot_result_time((F_1_9_tS)/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')

# ## Plot Cumulative Scrap surplus
# y_dict = {
#         'name': 'Cumulative Global scrap surplus',
#         'aspect': 'Scenario',
#         'unit': 'Mt Al'
#         }
# cf.plot_result_time(np.cumsum(F_1_9_tS, axis=0)/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, show = 'no', stack='no')
        
# ### Single scenario plots
# for scenario in range(2):

#     Mylog.info("Plotting results for scenario " + str(scenario))
#     ## Car Stock per region
#     y_dict = {
#             'name': 'Car Stock',
#             'aspect': 'Region',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(S_trS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Car inflows per region
#     y_dict = {
#             'name': 'Car Inflows',
#             'aspect': 'Region',
#             'unit': 'cars/yr'
#             }
#     cf.plot_result_time_scenario(I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     # Car outflows per region
#     y_dict = {
#             'name': 'Car Outflows',
#             'aspect': 'Region',
#             'unit': 'cars/yr'
#             }
#     cf.plot_result_time_scenario(O_trS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Stock by powertrain
#     y_dict = {
#             'name': 'Car Stock',
#             'aspect': 'Powertrain',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(S_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Car inflows by powertrain
#     y_dict = {
#             'name': 'Car inflows',
#             'aspect': 'Powertrain',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Car outflows by powertrain
#     y_dict = {
#             'name': 'Car Outflows',
#             'aspect': 'Powertrain',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(O_tpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Stock by segment
#     y_dict = {
#             'name': 'Car Stock',
#             'aspect': 'Segment',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(S_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Car inflows by segment
#     y_dict = {
#             'name': 'Car Inflows',
#             'aspect': 'Segment',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Car outflows by segment
#     y_dict = {
#             'name': 'Car Outflows',
#             'aspect': 'Segment',
#             'unit': 'cars'
#             }
#     cf.plot_result_time_scenario(O_tsS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
#     ## Plot Al Stock per component
#     y_dict = {
#             'name': 'Al Stock',
#             'aspect': 'Component',
#             'unit': 'Mt'
#             }
#     cf.plot_result_time_scenario(Al_stock_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Plot Al inflows per conponent
#     y_dict = {
#             'name': 'Al Inflows',
#             'aspect': 'Component',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Al_inflow_czS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
#     ## Plot Al outflows per conponent
#     y_dict = {
#             'name': 'Al Outflows',
            
#             'aspect': 'Component',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Al_outflow_tzS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
    
#     # Aluminium stock per region
#     y_dict = {
#             'name': 'Al stock',
#             'aspect': 'Region',
#             'unit': 'Mt'
#             }
#     cf.plot_result_time_scenario(Al_stock_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     # Aluminium inflows per region
#     y_dict = {
#             'name': 'Al Inflows',
#             'aspect': 'Region',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Al_inflow_crS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     # Aluminium outflows per region
#     y_dict = {
#             'name': 'Al Outflows',
#             'aspect': 'Region',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Al_outflow_trS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     ## Plot Al Alloys inflows 
#     y_dict = {
#             'name': 'Al Inflows',
#             'aspect': 'Alloy',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Alloys_inflow_caS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
#     ## Plot Average Al content in inflows per region
#     y_dict = {
#             'name': 'Average Al content',
#             'aspect': 'Region',
#             'unit': 'kg/car'
#             }
#     cf.plot_result_time_scenario(Al_inflow_crS / I_crS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
    
#     ## Plot Average Al content in inflows per powertrain
#     I_cpS = np.einsum('crpsS -> cpS', I_crpsS)
#     Al_inflow_cpS = np.einsum('crpsS -> cpS', Al_inflow_crpsS)
#     y_dict = {
#             'name': 'Average Al content',
#             'aspect': 'Powertrain',
#             'unit': 'kg/car'
#             }
#     cf.plot_result_time_scenario(Al_inflow_cpS / I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
    
#     ## Plot Average Al content in inflows per segment
#     I_csS = np.einsum('crpsS -> csS', I_crpsS)
#     Al_inflow_csS = np.einsum('crpsS -> csS', Al_inflow_crpsS)
#     y_dict = {
#             'name': 'Average Al content',
#             'aspect': 'Segment',
#             'unit': 'kg/car'
#             }
#     cf.plot_result_time_scenario(Al_inflow_csS / I_csS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='no')
    
#     ## Plot Al Alloys outflows
#     y_dict = {
#             'name': 'Al Outflows',
#             'aspect': 'Alloy',
#             'unit': 'Mt/yr'
#             }
#     cf.plot_result_time_scenario(Alloys_outflow_taS/10**9, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')
    
#     # Plot Ratio Outflows / Inflows
#     y_dict = {
#             'name': 'Ratio Outflows - Inflows',
#             'aspect': 'Alloy',
#             'unit': ''
#             }
#     cf.plot_result_time_scenario(Alloys_outflow_taS / Alloys_inflow_caS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no')
    
#     # Plot Ratio Outflows / Inflows for secondary castings per region
#     y_dict = {
#             'name': 'O-I Ratio for 2nd castings',
#             'aspect': 'Region',
#             'unit': ''
#             }
#     cf.plot_result_time_scenario(Alloys_outflow_traS[:,:,2] / Alloys_inflow_craS[:,:,2], y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no')


# end_time = time.time()
# Mylog.info("Time for plotting: ", end_time - start_time)



# # %% Custom Plots
# current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
# plot_dir = os.path.join('results', 'plots','custom', current_datetime)

# scenario = 0

# ## Car inflows by powertrain
# y_dict = {
#         'name': 'Global Car inflows',
#         'aspect': 'Powertrain',
#         'unit': 'cars'
#         }
# cf.plot_result_time_scenario(I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')

# scenario = 1

# ## Car inflows by powertrain
# y_dict = {
#         'name': 'Globlal Car inflows',
#         'aspect': 'Powertrain',
#         'unit': 'cars'
#         }
# cf.plot_result_time_scenario(I_cpS, y_dict, IndexTable, t_min= 100, t_max = 151, plot_dir=plot_dir, scenario=scenario, show = 'no', stack='yes')


# ## Plot Al inflows per scenario
# y_dict = {
#         'name': 'Global Al demand',
#         'aspect': 'Scenario',
#         'unit': 'Mt/year'
#         }
# t_min= 120
# t_max = 151
# category = IndexTable.Classification[y_dict['aspect']].Items[:2]
# array = Al_inflow_cS/10**9
# MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/2)) # select 10 colors from the 'Paired' color map.
# fig, ax = plt.subplots()
# ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
#            array[t_min:t_max,0],
#            color = MyColorCycle[0,:], linewidth = 2)
# ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
#            array[t_min:t_max,1],
#            color = MyColorCycle[1,:], linewidth = 2)
# ax.set_ylabel(y_dict['unit'],fontsize =16)
# fig.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
# ax.legend(category, loc='upper left',prop={'size':8})
# cf.mkdir_p(plot_dir)
# plot_path = plot_dir + '/' + y_dict['name'] +' by ' + y_dict['aspect']
# fig.savefig(plot_path, dpi = 400)    
# plt.show()
# plt.cla()
# plt.clf()
# plt.close(fig)
# Mylog.info("Saved to: " + plot_path)

# ## Plot Cumulative Carbon footprint
# y_dict = {
#         'name': 'Cumulative Carbon footprint of Al production',
#         'aspect': 'Scenario',
#         'unit': 'Gt CO2'
#         }
# t_min= 120
# t_max = 151
# category = IndexTable.Classification[y_dict['aspect']].Items[:2]
# array = np.cumsum(carbon_footprint_primary + carbon_footprint_secondary, axis=0)/10**12
# MyColorCycle = pylab.cm.Paired(np.arange(0,1,1/2)) # select 10 colors from the 'Paired' color map.
# fig, ax = plt.subplots()
# ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
#            array[t_min:t_max,0],
#            color = MyColorCycle[0,:], linewidth = 2)
# ax.plot(IndexTable['Classification']['Time'].Items[t_min:t_max],
#            array[t_min:t_max,1],
#            color = MyColorCycle[1,:], linewidth = 2)
# ax.set_ylabel(y_dict['unit'],fontsize =16)
# fig.suptitle(y_dict['name'] +' by ' + y_dict['aspect'])
# ax.legend(category, loc='upper left',prop={'size':8})
# cf.mkdir_p(plot_dir)
# plot_path = plot_dir + '/' + y_dict['name'] +' by ' + y_dict['aspect']
# fig.savefig(plot_path, dpi = 400)    
# plt.show()
# plt.cla()
# plt.clf()
# plt.close(fig)
# Mylog.info("Saved to: " + plot_path)










