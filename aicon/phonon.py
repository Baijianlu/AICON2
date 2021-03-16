# -*- coding: utf-8 -*-
# Copyright (C) 2020 Tao Fan
# All rights reserved.
#
# This file is part of AICON.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the AICON project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.

import os
import numpy as np
import scipy.constants
from scipy.integrate import quad
from aicon.tools import Get_GVD, calc_MFPS
import pymatgen as pmg
import pandas as pd

Planck = scipy.constants.hbar
Boltzm = scipy.constants.Boltzmann
atommass = scipy.constants.atomic_mass

def Get_Phonon(filepath, Temp, ifscale):
    '''For lattice thermal conductivity calculation '''
    kappa = Phonon(filepath)
    kappa.Get_Kappa(filepath, Temp, ifscale=ifscale)
    kappa.Output(Temp)

def t_Umklapp(grun,velo,Debye,mass,T):                          #relaxation time of umklapp process
    return (grun**2 * Boltzm**2 * T**3)/(mass * velo**2 * Debye * Planck) * np.exp(-Debye/(3*T))

def t_Normal(grun,velo,mass,vol,T):                             #relaxation time of normal process
    return (grun**2 * (Boltzm*1e23)**5 * (T*1e-2)**5 * (vol*1e30))/((mass*1e27) * (velo*1e-3)**5 * (Planck*1e34)**4) * 1e13

def t_Isotope(velo,vol,abund,T):                                #relaxation time of isotope scattering
    return ((vol*1e30) * (Boltzm*1e23)**4 * abund * (T*1e-2)**4)/(4 * np.pi * (Planck*1e34)**4 * (velo*1e-3)**3) * 1e13    

def constC(velo):
    return Boltzm**4/(2 * np.pi**2 * Planck**3 * velo)          #

def get_fun1(x,RT_N,RT_U,RT_ISO):
    return 1/(RT_N * x + RT_U * x**2 + RT_ISO * x**4) * x**4 * np.exp(x)/(np.exp(x)-1)**2

def get_fun2(x,RT_N,RT_U,RT_ISO):
    return RT_N/(RT_N + RT_U * x + RT_ISO * x**3) * x**4 * np.exp(x)/(np.exp(x)-1)**2

def get_fun3(x,RT_N,RT_U,RT_ISO):
    return RT_N * (RT_U + RT_ISO * x**2) /(RT_N + RT_U * x + RT_ISO * x**3) * x**6 * np.exp(x)/(np.exp(x)-1)**2 

def get_fun4(x,RT_N,RT_U,RT_ISO):
    return 1/(RT_N + RT_U + RT_ISO * x**2) * x**2 * np.exp(x)/(np.exp(x)-1)**2

def get_fun5(x,RT_N,RT_U,RT_ISO):
    return RT_N/(RT_N + RT_U + RT_ISO * x**2) * x**4 * np.exp(x)/(np.exp(x)-1)**2

def get_fun6(x,RT_N,RT_U,RT_ISO):
    return RT_N * (RT_U + RT_ISO * x**2)/(RT_N + RT_U + RT_ISO * x**2) * x**6 * np.exp(x)/(np.exp(x)-1)**2


class Phonon(object):
    ''' Phonon related properties class. '''
    
    def __init__(self, filepath):
        self.struct = pmg.Structure.from_file(filepath + 'POSCAR')
        self.M_avg = 0.0
        
        for ele in self.struct.symbol_set:
            self.M_avg = self.M_avg + pmg.Element(ele).atomic_mass * self.struct.composition.get_atomic_fraction(ele)
    
        self.M_avg = atommass * self.M_avg             
        self.V_avg = self.struct.volume/self.struct.composition.num_atoms * 1e-30  
        
        self.prims = self.struct.get_primitive_structure()
        self.Vol = self.prims.volume * 1e-30                                  
        self.p = self.prims.composition.num_atoms
        
    def Get_Para(self, filepath):
        (self.gruneisen, self.velocity, self.DebyeT, self.freq, self.optic_base) = Get_GVD(filepath)
        self.velocity = self.velocity * 1e2
        self.abund = calc_MFPS(list(self.struct.symbol_set))
        self.ADebye = self.DebyeT[2]                                          
        self.ODebye = self.DebyeT[3]
        print(self.gruneisen)
        print(self.velocity)
        print(self.DebyeT)
        print(self.freq)
        
    def HeatCapacity(self, ADebye, ODebye, T, struct):                    #function to calculate heat capacity
        N = 1                                                             # number of primitive cell
        prims = struct.get_primitive_structure()
        Vol = prims.volume * 1e-30                                        # primitive cell volume  
        p = prims.composition.num_atoms                                   # atom number in primitive cell
        fun = lambda x: x**4 * np.exp(x)/(np.exp(x)-1)**2
        Cv_aco = 9 * N/Vol * Boltzm * (T/ADebye)**3 * quad(fun,0,ADebye/T)[0]
        Cv_opt = (3*p-3) * N/Vol * Boltzm * (ODebye/T)**2 * np.exp(ODebye/T)/(np.exp(ODebye/T)-1)**2
        return Cv_aco, Cv_opt    


    def Output(self, Temp):
        Kappa_dict={"Temp": Temp, "Kappa": self.avgkappa, \
                    "TA_N": self.relaxtime[:,0,0], "TA_U": self.relaxtime[:,0,1], "TA_ISO": self.relaxtime[:,0,2],\
                    "TA\'_N": self.relaxtime[:,1,0], "TA\'_U": self.relaxtime[:,1,1], "TA\'_ISO": self.relaxtime[:,1,2],\
                    "LA_N": self.relaxtime[:,2,0], "LA_U": self.relaxtime[:,2,1], "LA_ISO": self.relaxtime[:,2,2],\
                    "O_N": self.relaxtime[:,3,0], "O_U": self.relaxtime[:,3,1], "O_ISO": self.relaxtime[:,3,2]}
        Kappa_FILE = pd.DataFrame(Kappa_dict)
        Kappa_FILE.to_excel('Kappa.xlsx', index_label='index', merge_cells=False)
#        Kappa_FILE.to_csv('Kappa.csv', index_label='index')
        
    def Get_Kappa(self, filepath, Temp, ifscale=False):
        ''' 
        Calculate lattice thermal conductivity at given temperature.
        
        Parameters:
        ----------
        filepath: str
            The directory contain POSCAR, band.yaml and gruneisen.yaml files.
        Temp: list
            The list of temperature.
        ifscale: bool
            If True, the calculated Kappa will be multiplied with a predefined factor.
            The default value is False.
        '''
        
        self.kappa = np.zeros((len(Temp), 4))
        self.avgkappa = np.zeros(len(Temp))
        self.relaxtime = np.zeros((len(Temp), 4, 3))
        self.ratio =  np.zeros(len(Temp))
        
        self.Get_Para(filepath)

        if ifscale:
            scale = 0.60371
        else:
            scale = 1.0

        for k in np.arange(len(Temp)):
            T = Temp[k]
            for branch in np.arange(4):                                          
                if branch == 0 or branch == 1:                                   # two transverse acoustic branches
                    coef_TU = t_Umklapp(self.gruneisen[branch],self.velocity[branch],self.DebyeT[branch],self.M_avg,T)
                    coef_TN = t_Normal(self.gruneisen[branch], self.velocity[branch], self.M_avg, self.V_avg, T)
                    coef_TISO = t_Isotope(self.velocity[branch], self.V_avg, self.abund, T)
                    C_T = constC(self.velocity[branch])
                    IT_1 = C_T * T**3 * quad(get_fun1, 0.0, self.DebyeT[branch]/T, args=(coef_TN,coef_TU,coef_TISO))[0]
                    BettaT_1 = quad(get_fun2, 0.0, self.DebyeT[branch]/T, args=(coef_TN,coef_TU,coef_TISO))[0]
                    BettaT_2 = quad(get_fun3, 0.0, self.DebyeT[branch]/T, args=(coef_TN,coef_TU,coef_TISO))[0]
                    IT_2 = C_T * T**3 * BettaT_1**2/BettaT_2
                    self.kappa[k, branch] = 1/3 * (IT_1 + IT_2)
                    self.relaxtime[k, branch, 0] = 1 / (coef_TN * self.DebyeT[branch]/T)
                    self.relaxtime[k, branch, 1] = 1 / (coef_TU * (self.DebyeT[branch]/T)**2)
                    self.relaxtime[k, branch, 2] = 1 / (coef_TISO * (self.DebyeT[branch]/T)**4)
                elif branch == 2:                                               # one longitudinal acoustic branch
                    coef_LU = t_Umklapp(self.gruneisen[branch],self.velocity[branch],self.DebyeT[branch],self.M_avg,T)
                    coef_LN = t_Normal(self.gruneisen[branch],self.velocity[branch],self.M_avg,self.V_avg,T)
                    coef_LISO = t_Isotope(self.velocity[branch],self.V_avg,self.abund,T)
                    C_L = constC(self.velocity[branch])
                    IL_1 = C_L * T**3 * quad(get_fun4, 0.0, self.DebyeT[branch]/T, args=(coef_LN,coef_LU,coef_LISO))[0]
                    BettaL_1 = quad(get_fun5, 0.0, self.DebyeT[branch]/T, args=(coef_LN,coef_LU,coef_LISO))[0]
                    BettaL_2 = quad(get_fun6, 0.0, self.DebyeT[branch]/T, args=(coef_LN,coef_LU,coef_LISO))[0]
                    IL_2 = C_L * T**3 * BettaL_1**2/BettaL_2
                    self.kappa[k, branch] = 1/3 * (IL_1 + IL_2)
                    self.relaxtime[k, branch, 0] = 1 / (coef_LN * (self.DebyeT[branch]/T)**2)
                    self.relaxtime[k, branch, 1] = 1 / (coef_LU * (self.DebyeT[branch]/T)**2)
                    self.relaxtime[k, branch, 2] = 1 / (coef_LISO * (self.DebyeT[branch]/T)**4)
                else:                                                              # optical branches
                    coef_OU = t_Umklapp(self.gruneisen[branch],self.velocity[branch],self.DebyeT[branch],self.M_avg,T)
                    coef_ON = t_Normal(self.gruneisen[branch],self.velocity[branch],self.M_avg,self.V_avg,T)
                    coef_OISO = t_Isotope(self.velocity[branch],self.V_avg,self.abund,T)
                    Normal_time = 1.0 / (coef_ON * (self.DebyeT[branch]/T)**2)
                    Resist_time = 1.0 / (coef_OU * (self.DebyeT[branch]/T)**2 + coef_OISO * (self.DebyeT[branch]/T)**4)
                    Total_time = 1.0 / (1.0/Normal_time + 1.0/Resist_time)
                    IO_1 = (3 * self.p -3) * 1/self.Vol * Boltzm * (self.ODebye/T)**2 * np.exp(self.ODebye/T)/(np.exp(self.ODebye/T)-1)**2 * self.velocity[branch]**2 * Total_time
                    IO_2 = (3 * self.p -3) * 1/self.Vol * Boltzm * (self.ODebye/T)**2 * np.exp(self.ODebye/T)/(np.exp(self.ODebye/T)-1)**2 * self.velocity[branch]**2 * Total_time/Normal_time * Resist_time
                    self.kappa[k, branch] = 1/3 * (IO_1 + IO_2)
                    self.relaxtime[k, branch, 0] = 1 / (coef_ON * (self.DebyeT[branch]/T)**2)
                    self.relaxtime[k, branch, 1] = 1 / (coef_OU * (self.DebyeT[branch]/T)**2)
                    self.relaxtime[k, branch, 2] = 1 / (coef_OISO * (self.DebyeT[branch]/T)**4)
            
            self.avgkappa[k] = scale * np.sum(self.kappa[k, :]) 
        
                
