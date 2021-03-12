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
from pymatgen.io.vasp import BSVasprun
from pymatgen.electronic_structure.core import Spin
from aicon.band import Band
import scipy.constants
import pandas as pd

EBoltzm = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
C_e = scipy.constants.e

def Get_Electron(filepath, Temp, dope, mode, ifSB):
    '''For electron transport properties calculation '''
    Compound = Electron()
    Compound.Get_bandstru(filepath + "equi/")
    Compound.Get_values(filepath, Temp, dope, mode, ifSB=ifSB)
    Compound.Output(Temp, dope, mode)

def find_mu_doping(band, doping, mu, Temp):
    '''
    Find the reduced chemical potential corresponding to the given carrier concentration.
    For carrier from one band.
    
    Parameters
    ----------
    band: Band object
    doping: float
        The specified carrier concentration, in the unit of cm-3.
    mu: list
        The list of chemical potential, unit is eV. Should have default values.
    Temp: list
        The list of Temperature, unit is K.
        
    Returns
    -------
    mu_x: list
        The list of reduced chemical potential corresponding to given doping level and temperature.
    '''
    mu_x = np.zeros((len(Temp), len(doping)))
    delta = np.zeros((len(doping), len(mu)))
    
    for i, T in enumerate(Temp):
        X = mu / (EBoltzm * T)
        for j, x in enumerate(X):
            N = band.Density(x, T) / 1e6
            for k, n in enumerate(doping):
                delta[k,j] = N - n
        delta = np.abs(delta)
        mu_x[i] = np.array([X[ind] for ind in np.argmin(delta, axis=1)])    
    
    return mu_x

def find_mu_doping2(band1, band2, doping, mu, Temp):
    '''
    Find the reduced chemical potential corresponding to the given carrier concentration.
    For carrier from two bands.
    
    Parameters
    ----------
    band: Band object
    doping: float
        The specified carrier concentration, in the unit of cm-3.
    mu: list
        The list of chemical potential, unit is eV. Should have default values.
    Temp: list
        The list of Temperature, unit is K.
        
    Returns
    -------
    mu_x: list
        The list of reduced chemical potential corresponding to given doping level and temperature.    
    '''
    mu_x = np.zeros((len(Temp), len(doping)))
    delta = np.zeros((len(doping), len(mu)))
    
    for i, T in enumerate(Temp):
        X = mu / (EBoltzm * T)
        gap = (band2.bandgap - band1.bandgap)/(EBoltzm*T)                    
        for j, x in enumerate(X):
            N_1 = band1.Density(x, T) / 1e6
            N_2 = band2.Density(x - gap, T) / 1e6
            for k, n in enumerate(doping):
                delta[k,j] = N_1 + N_2 - n
        delta = np.abs(delta)
        mu_x[i] = np.array([X[ind] for ind in np.argmin(delta, axis=1)])    
    
    return mu_x

class Electron(object):
    '''Electronic transport properties calculation class'''
    def Get_bandstru(self, filepath):
        '''Obtain band structure'''
        
        vaspband = BSVasprun(filepath + "/vasprun.xml")
        self.engband = vaspband.get_band_structure(kpoints_filename=filepath+"/KPOINTS",line_mode=True)
        self.bandgap = self.engband.get_band_gap()['energy']
    
    def Get_CBM(self):
        '''Find CBM and instantiate a Band object'''
        
        coord = self.engband.get_cbm()['kpoint'].frac_coords
        deg = self.engband.get_kpoint_degeneracy(coord)
        cbmbnd = np.min(self.engband.get_cbm()['band_index'][Spin.up])
        cbmkpt = self.engband.get_cbm()['kpoint_index'][0]
        self.CBM = Band(self.bandgap, deg, isCBM = True, bndindex=cbmbnd, kptindex=cbmkpt)
        
    def Get_VBM(self):
        '''Find VBM and instantiate a Band object'''
        
        coord = self.engband.get_vbm()['kpoint'].frac_coords
        deg = self.engband.get_kpoint_degeneracy(coord)
        vbmbnd = np.max(self.engband.get_vbm()['band_index'][Spin.up])
        vbmkpt = self.engband.get_vbm()['kpoint_index'][0]
        self.VBM = Band(self.bandgap, deg, isVBM = True, bndindex=vbmbnd, kptindex=vbmkpt)
        
    def Get_SB(self):
        '''Find CSB and VSB, then instantiate a Band object'''
        
        CSB_list = list()
        VSB_list = list()
        cbmbnd = np.min(self.engband.get_cbm()['band_index'][Spin.up])
        vbmbnd = np.max(self.engband.get_vbm()['band_index'][Spin.up])
        cbmeng = self.engband.get_cbm()['energy']
        vbmeng = self.engband.get_vbm()['energy']
        for i in np.arange(1, len(self.engband.kpoints)-1):
            if (self.engband.kpoints[i].frac_coords == self.engband.kpoints[i-1].frac_coords).all() or (self.engband.kpoints[i].frac_coords == self.engband.kpoints[i+1].frac_coords).all():
                if (self.engband.bands[Spin.up][cbmbnd,i] <= self.engband.bands[Spin.up][cbmbnd,i-2]) and (self.engband.bands[Spin.up][cbmbnd,i] <= self.engband.bands[Spin.up][cbmbnd,i+2]) and (np.abs(cbmeng - self.engband.bands[Spin.up][cbmbnd,i]) < 0.2):
                    if i not in self.engband.get_cbm()['kpoint_index']:
                        CSB_list.append(i)
            
                if (self.engband.bands[Spin.up][vbmbnd,i] >= self.engband.bands[Spin.up][vbmbnd,i-2]) and (self.engband.bands[Spin.up][vbmbnd,i] >= self.engband.bands[Spin.up][vbmbnd,i+2]) and (np.abs(vbmeng - self.engband.bands[Spin.up][vbmbnd,i]) < 0.2):
                    if i not in self.engband.get_vbm()['kpoint_index']:
                        VSB_list.append(i)
                        
            else:
                if (self.engband.bands[Spin.up][cbmbnd,i] <= self.engband.bands[Spin.up][cbmbnd,i-1]) and (self.engband.bands[Spin.up][cbmbnd,i] <= self.engband.bands[Spin.up][cbmbnd,i+1]) and (np.abs(cbmeng - self.engband.bands[Spin.up][cbmbnd,i]) < 0.2):
                    if i not in self.engband.get_cbm()['kpoint_index']:
                        CSB_list.append(i)
            
                if (self.engband.bands[Spin.up][vbmbnd,i] >= self.engband.bands[Spin.up][vbmbnd,i-1]) and (self.engband.bands[Spin.up][vbmbnd,i] >= self.engband.bands[Spin.up][vbmbnd,i+1]) and (np.abs(vbmeng - self.engband.bands[Spin.up][vbmbnd,i]) < 0.2):
                    if i not in self.engband.get_vbm()['kpoint_index']:
                        VSB_list.append(i)                

        CSB = None
        VSB = None
        if len(CSB_list) != 0:
            CSB = CSB_list[0]
            for i in np.arange(1, len(CSB_list)):
                if self.engband.bands[Spin.up][cbmbnd,CSB_list[i]] < self.engband.bands[Spin.up][cbmbnd,CSB]:
                    CSB = CSB_list[i]
                    
        if len(VSB_list) != 0:
            VSB = VSB_list[0]
            for i in np.arange(1, len(VSB_list)):
                if self.engband.bands[Spin.up][vbmbnd,VSB_list[i]] > self.engband.bands[Spin.up][vbmbnd,VSB]:
                    VSB = VSB_list[i]
        
        if CSB is not None:
            coord = self.engband.kpoints[CSB].frac_coords
            deg = self.engband.get_kpoint_degeneracy(coord)
            self.CSB = Band(self.bandgap + np.abs(self.engband.bands[Spin.up][cbmbnd,CSB] - cbmeng), deg, isCSB = True, bndindex=cbmbnd, kptindex=CSB)
        if VSB is not None:
            coord = self.engband.kpoints[VSB].frac_coords
            deg = self.engband.get_kpoint_degeneracy(coord)
            self.VSB = Band(self.bandgap + np.abs(self.engband.bands[Spin.up][vbmbnd,VSB] - vbmeng), deg, isVSB = True, bndindex=vbmbnd, kptindex=VSB)     

    def Get_values(self, filepath, Temp, doping, mode, ifSB=True):
        '''
        Calculate electronic transport properties. The results are either a function of chemical potential and temperature or a function of 
        carrier concentration and temperature.
        
        Parameters:
        ----------
        filepath: str
            
        Temp: list
            The specified temperatures, the unit is K.
        doping: list
            The specified carrier concentration, the unit is cm-3. Used in 'doping' mode.
        mode: str
            either 'standard' or 'doping'. In standard mode, the results are a function of chemical potential and temperature, while in 
            doping mode, the results are a function of specified carrier concentration and temperature.
        ifSB: bool
            if the second bands (CSB or VSB) are included in the calculation. Sometimes users may not want to consider them in the calculation 
            even they exist. The default value is True.
            
        The results are stored in the self.data attribute.
        '''
        self.data = dict()
        
        if mode == 'standard':
            mu = np.arange(-self.bandgap/2, -self.bandgap/2 + 1.0, 0.002)                #start from the middle of the gap to 1.0 eV higher place, with the stepsize 0.002 eV.
            self.mu = mu
            self.Get_CBM()
            self.Get_VBM()
            if ifSB:
                self.Get_SB()
                
            mux_array=list()
            for T in Temp:
                temporary = [ele/(EBoltzm*T) for ele in mu]
                mux_array.append(temporary)
            mux_array = np.array(mux_array)
            (CBM_AcoRelaxT, CBM_OptRelaxT, CBM_ImpRelaxT, CBM_TotalRelaxT, CBM_Density, CBM_Mobility,\
             CBM_Elcond, CBM_Seebeck, CBM_Lorenz, CBM_Ekappa, CBM_Hallfactor, CBM_Hallcoeff, CBM_PF) = self.CBM.Get_transport_para(filepath, mux_array, Temp)
            (VBM_AcoRelaxT, VBM_OptRelaxT, VBM_ImpRelaxT, VBM_TotalRelaxT, VBM_Density, VBM_Mobility,\
             VBM_Elcond, VBM_Seebeck, VBM_Lorenz, VBM_Ekappa, VBM_Hallfactor, VBM_Hallcoeff, VBM_PF) = self.VBM.Get_transport_para(filepath, mux_array, Temp) 
            self.data['CBM'] = {'TotalRelaxT':CBM_TotalRelaxT, 'AcoRelaxT':CBM_AcoRelaxT, 'OptRelaxT':CBM_OptRelaxT, 'ImpRelaxT':CBM_ImpRelaxT, \
                                 'Mobility':CBM_Mobility, 'Density':CBM_Density, 'Elcond':CBM_Elcond, 'Seebeck':CBM_Seebeck, 'Lorenz':CBM_Lorenz, \
                                 'Ekappa':CBM_Ekappa, 'Hallcoeff':CBM_Hallcoeff, 'PF':CBM_PF}
            self.data['VBM'] = {'TotalRelaxT':VBM_TotalRelaxT, 'AcoRelaxT':VBM_AcoRelaxT, 'OptRelaxT':VBM_OptRelaxT, 'ImpRelaxT':VBM_ImpRelaxT, \
                                 'Mobility':VBM_Mobility, 'Density':VBM_Density, 'Elcond':VBM_Elcond, 'Seebeck':VBM_Seebeck, 'Lorenz':VBM_Lorenz, \
                                 'Ekappa':VBM_Ekappa, 'Hallcoeff':VBM_Hallcoeff, 'PF':VBM_PF}
            
            if hasattr(self, 'CSB'):
                cmux_array=list()
                gap_csb = self.CSB.bandgap - self.CBM.bandgap
                for i, T in enumerate(Temp):
                    temporary = mux_array[i,:] - gap_csb/(EBoltzm*T)
                    cmux_array.append(temporary)
                cmux_array = np.array(cmux_array)
                (CSB_AcoRelaxT, CSB_OptRelaxT, CSB_ImpRelaxT, CSB_TotalRelaxT, CSB_Density, CSB_Mobility,\
                 CSB_Elcond, CSB_Seebeck, CSB_Lorenz, CSB_Ekappa, CSB_Hallfactor, CSB_Hallcoeff, CSB_PF) = self.CSB.Get_transport_para(filepath, cmux_array, Temp)
                TCB_Density = CBM_Density + CSB_Density
                TCB_Elcond = CBM_Elcond + CSB_Elcond
                TCB_Mobility = TCB_Elcond / (C_e * TCB_Density)
                TCB_Seebeck = (CBM_Seebeck * CBM_Elcond + CSB_Seebeck * CSB_Elcond ) / TCB_Elcond
                TCB_Lorenz = (CBM_Lorenz * CBM_Elcond + CSB_Lorenz * CSB_Elcond) / TCB_Elcond
                TCB_Ekappa = np.array([TCB_Lorenz[i] * TCB_Elcond[i] * T for i, T in enumerate(Temp)]) 
                TCB_Hallcoeff = (CBM_Hallfactor * CBM_Mobility * CBM_Elcond + CSB_Hallfactor * CSB_Mobility * CSB_Elcond) / (TCB_Elcond)**2
                TCB_PF = TCB_Seebeck**2 * TCB_Elcond
                self.data['CSB'] = {'TotalRelaxT':CSB_TotalRelaxT, 'AcoRelaxT':CSB_AcoRelaxT, 'OptRelaxT':CSB_OptRelaxT, 'ImpRelaxT':CSB_ImpRelaxT, \
                                     'Mobility':CSB_Mobility, 'Density':CSB_Density, 'Elcond':CSB_Elcond, 'Seebeck':CSB_Seebeck, 'Lorenz':CSB_Lorenz, \
                                     'Ekappa':CSB_Ekappa, 'Hallcoeff':CSB_Hallcoeff, 'PF':CSB_PF}
                self.data['TCB'] = {'Mobility':TCB_Mobility, 'Density':TCB_Density, 'Elcond':TCB_Elcond, 'Seebeck':TCB_Seebeck, 'Lorenz':TCB_Lorenz, \
                                     'Ekappa':TCB_Ekappa, 'Hallcoeff':TCB_Hallcoeff, 'PF':TCB_PF}
                
            if hasattr(self, 'VSB'):
                vmux_array=list()
                gap_vsb = self.VSB.bandgap - self.VBM.bandgap
                for i, T in enumerate(Temp):
                    temporary = mux_array[i,:] - gap_vsb/(EBoltzm*T)
                    vmux_array.append(temporary)
                vmux_array = np.array(vmux_array)  
                (VSB_AcoRelaxT, VSB_OptRelaxT, VSB_ImpRelaxT, VSB_TotalRelaxT, VSB_Density, VSB_Mobility,\
                 VSB_Elcond, VSB_Seebeck, VSB_Lorenz, VSB_Ekappa, VSB_Hallfactor, VSB_Hallcoeff, VSB_PF) = self.VSB.Get_transport_para(filepath, vmux_array, Temp)
                TVB_Density = VBM_Density + VSB_Density
                TVB_Elcond = VBM_Elcond + VSB_Elcond
                TVB_Mobility = TVB_Elcond / (C_e * TVB_Density)
                TVB_Seebeck = (VBM_Seebeck * VBM_Elcond + VSB_Seebeck * VSB_Elcond ) / TVB_Elcond
                TVB_Lorenz = (VBM_Lorenz * VBM_Elcond + VSB_Lorenz * VSB_Elcond) / TVB_Elcond
                TVB_Ekappa = np.array([TVB_Lorenz[i] * TVB_Elcond[i] * T for i, T in enumerate(Temp)]) 
                TVB_Hallcoeff = (VBM_Hallfactor * VBM_Mobility * VBM_Elcond + VSB_Hallfactor * VSB_Mobility * VSB_Elcond) / (TVB_Elcond)**2
                TVB_PF = TVB_Seebeck**2 * TVB_Elcond            
                self.data['VSB'] = {'TotalRelaxT':VSB_TotalRelaxT, 'AcoRelaxT':VSB_AcoRelaxT, 'OptRelaxT':VSB_OptRelaxT, 'ImpRelaxT':VSB_ImpRelaxT, \
                                     'Mobility':VSB_Mobility, 'Density':VSB_Density, 'Elcond':VSB_Elcond, 'Seebeck':VSB_Seebeck, 'Lorenz':VSB_Lorenz, \
                                     'Ekappa':VSB_Ekappa, 'Hallcoeff':VSB_Hallcoeff, 'PF':VSB_PF}
                self.data['TVB'] = {'Mobility':TVB_Mobility, 'Density':TVB_Density, 'Elcond':TVB_Elcond, 'Seebeck':TVB_Seebeck, 'Lorenz':TVB_Lorenz, \
                                     'Ekappa':TVB_Ekappa, 'Hallcoeff':TVB_Hallcoeff, 'PF':TVB_PF}                


        if mode == 'doping':
            mu = np.arange(-self.bandgap/2, -self.bandgap/2 + 1.0, 0.0005)
            self.Get_CBM()
            self.Get_VBM()
            if ifSB:
                self.Get_SB()
            
            self.CBM.Get_carridensity(filepath)
            self.VBM.Get_carridensity(filepath)
            if hasattr(self, 'CSB'):
                gap_csb = self.CSB.bandgap - self.CBM.bandgap
                self.CSB.Get_carridensity(filepath)
                cmux_array = find_mu_doping2(self.CBM, self.CSB, doping, mu, Temp)
                csmux_array=list()
                for i, T in enumerate(Temp):
                    temporary = cmux_array[i,:] - gap_csb/(EBoltzm*T)
                    csmux_array.append(temporary)
                csmux_array = np.array(csmux_array)
                (CBM_AcoRelaxT, CBM_OptRelaxT, CBM_ImpRelaxT, CBM_TotalRelaxT, CBM_Density, CBM_Mobility,\
                 CBM_Elcond, CBM_Seebeck, CBM_Lorenz, CBM_Ekappa, CBM_Hallfactor, CBM_Hallcoeff, CBM_PF) = self.CBM.Get_transport_para(filepath, cmux_array, Temp)
                (CSB_AcoRelaxT, CSB_OptRelaxT, CSB_ImpRelaxT, CSB_TotalRelaxT, CSB_Density, CSB_Mobility,\
                 CSB_Elcond, CSB_Seebeck, CSB_Lorenz, CSB_Ekappa, CSB_Hallfactor, CSB_Hallcoeff, CSB_PF) = self.CSB.Get_transport_para(filepath, csmux_array, Temp)
                TCB_Density = CBM_Density + CSB_Density
                TCB_Elcond = CBM_Elcond + CSB_Elcond
                TCB_Mobility = TCB_Elcond / (C_e * TCB_Density)
                TCB_Seebeck = (CBM_Seebeck * CBM_Elcond + CSB_Seebeck * CSB_Elcond ) / TCB_Elcond
                TCB_Lorenz = (CBM_Lorenz * CBM_Elcond + CSB_Lorenz * CSB_Elcond) / TCB_Elcond
                TCB_Ekappa = np.array([TCB_Lorenz[i] * TCB_Elcond[i] * T for i, T in enumerate(Temp)]) 
                TCB_Hallcoeff = (CBM_Hallfactor * CBM_Mobility * CBM_Elcond + CSB_Hallfactor * CSB_Mobility * CSB_Elcond) / (TCB_Elcond)**2
                TCB_PF = TCB_Seebeck**2 * TCB_Elcond
                self.data['CBM'] = {'TotalRelaxT':CBM_TotalRelaxT, 'AcoRelaxT':CBM_AcoRelaxT, 'OptRelaxT':CBM_OptRelaxT, 'ImpRelaxT':CBM_ImpRelaxT, \
                                     'Mobility':CBM_Mobility, 'Density':CBM_Density, 'Elcond':CBM_Elcond, 'Seebeck':CBM_Seebeck, 'Lorenz':CBM_Lorenz, \
                                     'Ekappa':CBM_Ekappa, 'Hallcoeff':CBM_Hallcoeff, 'PF':CBM_PF}
                self.data['CSB'] = {'TotalRelaxT':CSB_TotalRelaxT, 'AcoRelaxT':CSB_AcoRelaxT, 'OptRelaxT':CSB_OptRelaxT, 'ImpRelaxT':CSB_ImpRelaxT, \
                                     'Mobility':CSB_Mobility, 'Density':CSB_Density, 'Elcond':CSB_Elcond, 'Seebeck':CSB_Seebeck, 'Lorenz':CSB_Lorenz, \
                                     'Ekappa':CSB_Ekappa, 'Hallcoeff':CSB_Hallcoeff, 'PF':CSB_PF}
                self.data['TCB'] = {'Mobility':TCB_Mobility, 'Density':TCB_Density, 'Elcond':TCB_Elcond, 'Seebeck':TCB_Seebeck, 'Lorenz':TCB_Lorenz, \
                                     'Ekappa':TCB_Ekappa, 'Hallcoeff':TCB_Hallcoeff, 'PF':TCB_PF}
                
            else:
                cmux_array = find_mu_doping(self.CBM, doping, mu, Temp)
                (CBM_AcoRelaxT, CBM_OptRelaxT, CBM_ImpRelaxT, CBM_TotalRelaxT, CBM_Density, CBM_Mobility,\
                 CBM_Elcond, CBM_Seebeck, CBM_Lorenz, CBM_Ekappa, CBM_Hallfactor, CBM_Hallcoeff, CBM_PF) = self.CBM.Get_transport_para(filepath, cmux_array, Temp)
                self.data['CBM'] = {'TotalRelaxT':CBM_TotalRelaxT, 'AcoRelaxT':CBM_AcoRelaxT, 'OptRelaxT':CBM_OptRelaxT, 'ImpRelaxT':CBM_ImpRelaxT, \
                                     'Mobility':CBM_Mobility, 'Density':CBM_Density, 'Elcond':CBM_Elcond, 'Seebeck':CBM_Seebeck, 'Lorenz':CBM_Lorenz, \
                                     'Ekappa':CBM_Ekappa, 'Hallcoeff':CBM_Hallcoeff, 'PF':CBM_PF}
                
            if hasattr(self, 'VSB'):
                gap_vsb = self.VSB.bandgap - self.VBM.bandgap
                self.VSB.Get_carridensity(filepath)
                vmux_array = find_mu_doping2(self.VBM, self.VSB, doping, mu, Temp)
                vsmux_array=list()
                for i, T in enumerate(Temp):
                    temporary = vmux_array[i,:] - gap_vsb/(EBoltzm*T)
                    vsmux_array.append(temporary)
                vsmux_array = np.array(vsmux_array)
                (VBM_AcoRelaxT, VBM_OptRelaxT, VBM_ImpRelaxT, VBM_TotalRelaxT, VBM_Density, VBM_Mobility,\
                 VBM_Elcond, VBM_Seebeck, VBM_Lorenz, VBM_Ekappa, VBM_Hallfactor, VBM_Hallcoeff, VBM_PF) = self.VBM.Get_transport_para(filepath, vmux_array, Temp)
                (VSB_AcoRelaxT, VSB_OptRelaxT, VSB_ImpRelaxT, VSB_TotalRelaxT, VSB_Density, VSB_Mobility,\
                 VSB_Elcond, VSB_Seebeck, VSB_Lorenz, VSB_Ekappa, VSB_Hallfactor, VSB_Hallcoeff, VSB_PF) = self.VSB.Get_transport_para(filepath, vsmux_array, Temp)
                TVB_Density = VBM_Density + VSB_Density
                TVB_Elcond = VBM_Elcond + VSB_Elcond
                TVB_Mobility = TVB_Elcond / (C_e * TVB_Density)
                TVB_Seebeck = (VBM_Seebeck * VBM_Elcond + VSB_Seebeck * VSB_Elcond ) / TVB_Elcond
                TVB_Lorenz = (VBM_Lorenz * VBM_Elcond + VSB_Lorenz * VSB_Elcond) / TVB_Elcond
                TVB_Ekappa = np.array([TVB_Lorenz[i] * TVB_Elcond[i] * T for i, T in enumerate(Temp)]) 
                TVB_Hallcoeff = (VBM_Hallfactor * VBM_Mobility * VBM_Elcond + VSB_Hallfactor * VSB_Mobility * VSB_Elcond) / (TVB_Elcond)**2
                TVB_PF = TVB_Seebeck**2 * TVB_Elcond 
                self.data['VBM'] = {'TotalRelaxT':VBM_TotalRelaxT, 'AcoRelaxT':VBM_AcoRelaxT, 'OptRelaxT':VBM_OptRelaxT, 'ImpRelaxT':VBM_ImpRelaxT, \
                                     'Mobility':VBM_Mobility, 'Density':VBM_Density, 'Elcond':VBM_Elcond, 'Seebeck':VBM_Seebeck, 'Lorenz':VBM_Lorenz, \
                                     'Ekappa':VBM_Ekappa, 'Hallcoeff':VBM_Hallcoeff, 'PF':VBM_PF}
                self.data['VSB'] = {'TotalRelaxT':VSB_TotalRelaxT, 'AcoRelaxT':VSB_AcoRelaxT, 'OptRelaxT':VSB_OptRelaxT, 'ImpRelaxT':VSB_ImpRelaxT, \
                                     'Mobility':VSB_Mobility, 'Density':VSB_Density, 'Elcond':VSB_Elcond, 'Seebeck':VSB_Seebeck, 'Lorenz':VSB_Lorenz, \
                                     'Ekappa':VSB_Ekappa, 'Hallcoeff':VSB_Hallcoeff, 'PF':VSB_PF}
                self.data['TVB'] = {'Mobility':TVB_Mobility, 'Density':TVB_Density, 'Elcond':TVB_Elcond, 'Seebeck':TVB_Seebeck, 'Lorenz':TVB_Lorenz, \
                                     'Ekappa':TVB_Ekappa, 'Hallcoeff':TVB_Hallcoeff, 'PF':TVB_PF}
                
            else:
                vmux_array = find_mu_doping(self.VBM, doping, mu, Temp)
                (VBM_AcoRelaxT, VBM_OptRelaxT, VBM_ImpRelaxT, VBM_TotalRelaxT, VBM_Density, VBM_Mobility,\
                 VBM_Elcond, VBM_Seebeck, VBM_Lorenz, VBM_Ekappa, VBM_Hallfactor, VBM_Hallcoeff, VBM_PF) = self.VBM.Get_transport_para(filepath, vmux_array, Temp)
                self.data['VBM'] = {'TotalRelaxT':VBM_TotalRelaxT, 'AcoRelaxT':VBM_AcoRelaxT, 'OptRelaxT':VBM_OptRelaxT, 'ImpRelaxT':VBM_ImpRelaxT, \
                                     'Mobility':VBM_Mobility, 'Density':VBM_Density, 'Elcond':VBM_Elcond, 'Seebeck':VBM_Seebeck, 'Lorenz':VBM_Lorenz, \
                                     'Ekappa':VBM_Ekappa, 'Hallcoeff':VBM_Hallcoeff, 'PF':VBM_PF}

            
#####################################################################################################################################################################                    
    def Output(self, Temp, doping, mode):
        ''' 
        Output the results as any file format users want. The results are firstly converted to a pandas.DataFrame object, so users can store it as any file
        format pandas supports. Also, the key parameters for each band are stored in Parameter file for checking. 
        '''
        shape = np.shape(self.data['CBM']['TotalRelaxT'])
        Temp_list = np.array([])
        
        if mode == 'standard':
            mu_list = np.array([])
        if mode == 'doping':
            dope_list= np.array([])
            
        CBM_density_list = np.array([])
        CBM_elcond_list = np.array([])
        CBM_seebeck_list = np.array([])
        CBM_mobility_list = np.array([])
        CBM_lorenz_list = np.array([])
        CBM_ekappa_list = np.array([])
        CBM_hallcoeff_list = np.array([])
        CBM_pf_list = np.array([])
        CBM_totalrelaxtime_list = np.array([])
        CBM_acorelaxtime_list = np.array([])
        CBM_optrelaxtime_list = np.array([])
        CBM_imprelaxtime_list = np.array([])
        VBM_density_list = np.array([])
        VBM_elcond_list = np.array([])
        VBM_seebeck_list = np.array([])
        VBM_mobility_list = np.array([])
        VBM_lorenz_list = np.array([])
        VBM_ekappa_list = np.array([])
        VBM_hallcoeff_list = np.array([])
        VBM_pf_list = np.array([])
        VBM_totalrelaxtime_list = np.array([])
        VBM_acorelaxtime_list = np.array([])
        VBM_optrelaxtime_list = np.array([])
        VBM_imprelaxtime_list = np.array([])
        
        if hasattr(self, 'CSB'):
            CSB_density_list = np.array([])
            CSB_elcond_list = np.array([])
            CSB_seebeck_list = np.array([])
            CSB_mobility_list = np.array([])
            CSB_lorenz_list = np.array([])
            CSB_ekappa_list = np.array([])
            CSB_hallcoeff_list = np.array([])
            CSB_pf_list = np.array([])
            CSB_totalrelaxtime_list = np.array([])
            CSB_acorelaxtime_list = np.array([])
            CSB_optrelaxtime_list = np.array([])
            CSB_imprelaxtime_list = np.array([])
            TCB_density_list = np.array([])
            TCB_elcond_list = np.array([])
            TCB_seebeck_list = np.array([])
            TCB_mobility_list = np.array([])
            TCB_lorenz_list = np.array([])
            TCB_ekappa_list = np.array([])
            TCB_hallcoeff_list = np.array([])
            TCB_pf_list = np.array([])
        if hasattr(self, 'VSB'):
            VSB_density_list = np.array([])
            VSB_elcond_list = np.array([])
            VSB_seebeck_list = np.array([])
            VSB_mobility_list = np.array([])
            VSB_lorenz_list = np.array([])
            VSB_ekappa_list = np.array([])
            VSB_hallcoeff_list = np.array([])
            VSB_pf_list = np.array([])
            VSB_totalrelaxtime_list = np.array([])
            VSB_acorelaxtime_list = np.array([])
            VSB_optrelaxtime_list = np.array([])
            VSB_imprelaxtime_list = np.array([])
            TVB_density_list = np.array([])
            TVB_elcond_list = np.array([])
            TVB_seebeck_list = np.array([])
            TVB_mobility_list = np.array([])
            TVB_lorenz_list = np.array([])
            TVB_ekappa_list = np.array([])
            TVB_hallcoeff_list = np.array([])
            TVB_pf_list = np.array([])
        
        for i in Temp:
            Temp_list=np.concatenate((Temp_list, np.repeat(i, shape[1])))
            if mode == 'doping':
                dope_list=np.concatenate((dope_list, doping))
            if mode == 'standard':
                mu_list=np.concatenate((mu_list, self.mu))
            
        for i in np.arange(shape[0]):
            CBM_density_list=np.concatenate((CBM_density_list, self.data['CBM']['Density'][i,:]))
            CBM_seebeck_list=np.concatenate((CBM_seebeck_list, self.data['CBM']['Seebeck'][i,:]))
            CBM_mobility_list = np.concatenate((CBM_mobility_list, self.data['CBM']['Mobility'][i,:]))
            CBM_elcond_list=np.concatenate((CBM_elcond_list, self.data['CBM']['Elcond'][i,:]))
            CBM_lorenz_list=np.concatenate((CBM_lorenz_list, self.data['CBM']['Lorenz'][i,:]))
            CBM_ekappa_list=np.concatenate((CBM_ekappa_list, self.data['CBM']['Ekappa'][i,:]))
            CBM_hallcoeff_list=np.concatenate((CBM_hallcoeff_list, self.data['CBM']['Hallcoeff'][i,:]))
            CBM_pf_list = np.concatenate((CBM_pf_list, self.data['CBM']['PF'][i,:]))
            CBM_totalrelaxtime_list = np.concatenate((CBM_totalrelaxtime_list, self.data['CBM']['TotalRelaxT'][i,:]))
            CBM_acorelaxtime_list = np.concatenate((CBM_acorelaxtime_list, self.data['CBM']['AcoRelaxT'][i,:]))
            CBM_optrelaxtime_list = np.concatenate((CBM_optrelaxtime_list, self.data['CBM']['OptRelaxT'][i,:]))
            CBM_imprelaxtime_list = np.concatenate((CBM_imprelaxtime_list, self.data['CBM']['ImpRelaxT'][i,:]))
            
            VBM_density_list=np.concatenate((VBM_density_list, self.data['VBM']['Density'][i,:]))
            VBM_seebeck_list=np.concatenate((VBM_seebeck_list, self.data['VBM']['Seebeck'][i,:]))
            VBM_mobility_list = np.concatenate((VBM_mobility_list, self.data['VBM']['Mobility'][i,:]))
            VBM_elcond_list=np.concatenate((VBM_elcond_list, self.data['VBM']['Elcond'][i,:]))
            VBM_lorenz_list=np.concatenate((VBM_lorenz_list, self.data['VBM']['Lorenz'][i,:]))
            VBM_ekappa_list=np.concatenate((VBM_ekappa_list, self.data['VBM']['Ekappa'][i,:]))
            VBM_hallcoeff_list=np.concatenate((VBM_hallcoeff_list, self.data['VBM']['Hallcoeff'][i,:]))
            VBM_pf_list = np.concatenate((VBM_pf_list, self.data['VBM']['PF'][i,:]))
            VBM_totalrelaxtime_list = np.concatenate((VBM_totalrelaxtime_list, self.data['VBM']['TotalRelaxT'][i,:]))
            VBM_acorelaxtime_list = np.concatenate((VBM_acorelaxtime_list, self.data['VBM']['AcoRelaxT'][i,:]))
            VBM_optrelaxtime_list = np.concatenate((VBM_optrelaxtime_list, self.data['VBM']['OptRelaxT'][i,:]))
            VBM_imprelaxtime_list = np.concatenate((VBM_imprelaxtime_list, self.data['VBM']['ImpRelaxT'][i,:]))
            
            if hasattr(self, 'CSB'):
                CSB_density_list=np.concatenate((CSB_density_list, self.data['CSB']['Density'][i,:]))
                CSB_seebeck_list=np.concatenate((CSB_seebeck_list, self.data['CSB']['Seebeck'][i,:]))
                CSB_mobility_list = np.concatenate((CSB_mobility_list, self.data['CSB']['Mobility'][i,:]))
                CSB_elcond_list=np.concatenate((CSB_elcond_list, self.data['CSB']['Elcond'][i,:]))
                CSB_lorenz_list=np.concatenate((CSB_lorenz_list, self.data['CSB']['Lorenz'][i,:]))
                CSB_ekappa_list=np.concatenate((CSB_ekappa_list, self.data['CSB']['Ekappa'][i,:]))
                CSB_hallcoeff_list=np.concatenate((CSB_hallcoeff_list, self.data['CSB']['Hallcoeff'][i,:]))
                CSB_pf_list = np.concatenate((CSB_pf_list, self.data['CSB']['PF'][i,:]))
                CSB_totalrelaxtime_list = np.concatenate((CSB_totalrelaxtime_list, self.data['CSB']['TotalRelaxT'][i,:]))
                CSB_acorelaxtime_list = np.concatenate((CSB_acorelaxtime_list, self.data['CSB']['AcoRelaxT'][i,:]))
                CSB_optrelaxtime_list = np.concatenate((CSB_optrelaxtime_list, self.data['CSB']['OptRelaxT'][i,:]))
                CSB_imprelaxtime_list = np.concatenate((CSB_imprelaxtime_list, self.data['CSB']['ImpRelaxT'][i,:]))
                TCB_density_list=np.concatenate((TCB_density_list, self.data['TCB']['Density'][i,:]))
                TCB_seebeck_list=np.concatenate((TCB_seebeck_list, self.data['TCB']['Seebeck'][i,:]))
                TCB_mobility_list = np.concatenate((TCB_mobility_list, self.data['TCB']['Mobility'][i,:]))
                TCB_elcond_list=np.concatenate((TCB_elcond_list, self.data['TCB']['Elcond'][i,:]))
                TCB_lorenz_list=np.concatenate((TCB_lorenz_list, self.data['TCB']['Lorenz'][i,:]))
                TCB_ekappa_list=np.concatenate((TCB_ekappa_list, self.data['TCB']['Ekappa'][i,:]))
                TCB_hallcoeff_list=np.concatenate((TCB_hallcoeff_list, self.data['TCB']['Hallcoeff'][i,:]))
                TCB_pf_list = np.concatenate((TCB_pf_list, self.data['TCB']['PF'][i,:]))
            
            if hasattr(self, 'VSB'):
                VSB_density_list=np.concatenate((VSB_density_list, self.data['VSB']['Density'][i,:]))
                VSB_seebeck_list=np.concatenate((VSB_seebeck_list, self.data['VSB']['Seebeck'][i,:]))
                VSB_mobility_list = np.concatenate((VSB_mobility_list, self.data['VSB']['Mobility'][i,:]))
                VSB_elcond_list=np.concatenate((VSB_elcond_list, self.data['VSB']['Elcond'][i,:]))
                VSB_lorenz_list=np.concatenate((VSB_lorenz_list, self.data['VSB']['Lorenz'][i,:]))
                VSB_ekappa_list=np.concatenate((VSB_ekappa_list, self.data['VSB']['Ekappa'][i,:]))
                VSB_hallcoeff_list=np.concatenate((VSB_hallcoeff_list, self.data['VSB']['Hallcoeff'][i,:]))
                VSB_pf_list = np.concatenate((VSB_pf_list, self.data['VSB']['PF'][i,:]))
                VSB_totalrelaxtime_list = np.concatenate((VSB_totalrelaxtime_list, self.data['VSB']['TotalRelaxT'][i,:]))
                VSB_acorelaxtime_list = np.concatenate((VSB_acorelaxtime_list, self.data['VSB']['AcoRelaxT'][i,:]))
                VSB_optrelaxtime_list = np.concatenate((VSB_optrelaxtime_list, self.data['VSB']['OptRelaxT'][i,:]))
                VSB_imprelaxtime_list = np.concatenate((VSB_imprelaxtime_list, self.data['VSB']['ImpRelaxT'][i,:]))
                TVB_density_list=np.concatenate((TVB_density_list, self.data['TVB']['Density'][i,:]))
                TVB_seebeck_list=np.concatenate((TVB_seebeck_list, self.data['TVB']['Seebeck'][i,:]))
                TVB_mobility_list = np.concatenate((TVB_mobility_list, self.data['TVB']['Mobility'][i,:]))
                TVB_elcond_list=np.concatenate((TVB_elcond_list, self.data['TVB']['Elcond'][i,:]))
                TVB_lorenz_list=np.concatenate((TVB_lorenz_list, self.data['TVB']['Lorenz'][i,:]))
                TVB_ekappa_list=np.concatenate((TVB_ekappa_list, self.data['TVB']['Ekappa'][i,:]))
                TVB_hallcoeff_list=np.concatenate((TVB_hallcoeff_list, self.data['TVB']['Hallcoeff'][i,:]))
                TVB_pf_list = np.concatenate((TVB_pf_list, self.data['TVB']['PF'][i,:]))
                    
        if mode == 'standard':
            CBM_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": CBM_density_list, "Seebeck": CBM_seebeck_list, \
                 "Mobility": CBM_mobility_list, "Elcond": CBM_elcond_list, "Lorenz": CBM_lorenz_list, "Ekappa": CBM_ekappa_list, \
                 "Hallcoeff": CBM_hallcoeff_list, "PF": CBM_pf_list, "TotalRelaxTime": CBM_totalrelaxtime_list, "AcoRelaxTime": CBM_acorelaxtime_list, \
                 "OptRelaxTime": CBM_optrelaxtime_list, "ImpRelaxTime": CBM_imprelaxtime_list}
            
            VBM_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": VBM_density_list, "Seebeck": VBM_seebeck_list, \
                 "Mobility": VBM_mobility_list, "Elcond": VBM_elcond_list, "Lorenz": VBM_lorenz_list, "Ekappa": VBM_ekappa_list, \
                 "Hallcoeff": VBM_hallcoeff_list, "PF": VBM_pf_list, "TotalRelaxTime": VBM_totalrelaxtime_list, "AcoRelaxTime": VBM_acorelaxtime_list, \
                 "OptRelaxTime": VBM_optrelaxtime_list, "ImpRelaxTime": VBM_imprelaxtime_list}
            
            if hasattr(self, 'CSB'):
                CSB_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": CSB_density_list, "Seebeck": CSB_seebeck_list, \
                     "Mobility": CSB_mobility_list, "Elcond": CSB_elcond_list, "Lorenz": CSB_lorenz_list, "Ekappa": CSB_ekappa_list, \
                     "Hallcoeff": CSB_hallcoeff_list, "PF": CSB_pf_list, "TotalRelaxTime": CSB_totalrelaxtime_list, "AcoRelaxTime": CSB_acorelaxtime_list, \
                     "OptRelaxTime": CSB_optrelaxtime_list, "ImpRelaxTime": CSB_imprelaxtime_list}
                
                TCB_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": TCB_density_list, "Seebeck": TCB_seebeck_list, \
                     "Mobility": TCB_mobility_list, "Elcond": TCB_elcond_list, "Lorenz": TCB_lorenz_list, "Ekappa": TCB_ekappa_list, \
                     "Hallcoeff": TCB_hallcoeff_list, "PF": TCB_pf_list}
                
            if hasattr(self, 'VSB'):
                VSB_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": VSB_density_list, "Seebeck": VSB_seebeck_list, \
                     "Mobility": VSB_mobility_list, "Elcond": VSB_elcond_list, "Lorenz": VSB_lorenz_list, "Ekappa": VSB_ekappa_list, \
                     "Hallcoeff": VSB_hallcoeff_list, "PF": VSB_pf_list, "TotalRelaxTime": VSB_totalrelaxtime_list, "AcoRelaxTime": VSB_acorelaxtime_list, \
                     "OptRelaxTime": VSB_optrelaxtime_list, "ImpRelaxTime": VSB_imprelaxtime_list}
                
                TVB_dict={"Temperature": Temp_list, "mu": mu_list, "Concentration": TVB_density_list, "Seebeck": TVB_seebeck_list, \
                     "Mobility": TVB_mobility_list, "Elcond": TVB_elcond_list, "Lorenz": TVB_lorenz_list, "Ekappa": TVB_ekappa_list, \
                     "Hallcoeff": TVB_hallcoeff_list, "PF": TVB_pf_list}
                
        if mode == 'doping':
            CBM_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": CBM_density_list, "Seebeck": CBM_seebeck_list, \
                 "Mobility": CBM_mobility_list, "Elcond": CBM_elcond_list, "Lorenz": CBM_lorenz_list, "Ekappa": CBM_ekappa_list, \
                 "Hallcoeff": CBM_hallcoeff_list, "PF": CBM_pf_list, "TotalRelaxTime": CBM_totalrelaxtime_list, "AcoRelaxTime": CBM_acorelaxtime_list, \
                 "OptRelaxTime": CBM_optrelaxtime_list, "ImpRelaxTime": CBM_imprelaxtime_list}
            
            VBM_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": VBM_density_list, "Seebeck": VBM_seebeck_list, \
                 "Mobility": VBM_mobility_list, "Elcond": VBM_elcond_list, "Lorenz": VBM_lorenz_list, "Ekappa": VBM_ekappa_list, \
                 "Hallcoeff": VBM_hallcoeff_list, "PF": VBM_pf_list, "TotalRelaxTime": VBM_totalrelaxtime_list, "AcoRelaxTime": VBM_acorelaxtime_list, \
                 "OptRelaxTime": VBM_optrelaxtime_list, "ImpRelaxTime": VBM_imprelaxtime_list}
            
            if hasattr(self, 'CSB'):
                CSB_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": CSB_density_list, "Seebeck": CSB_seebeck_list, \
                     "Mobility": CSB_mobility_list, "Elcond": CSB_elcond_list, "Lorenz": CSB_lorenz_list, "Ekappa": CSB_ekappa_list, \
                     "Hallcoeff": CSB_hallcoeff_list, "PF": CSB_pf_list, "TotalRelaxTime": CSB_totalrelaxtime_list, "AcoRelaxTime": CSB_acorelaxtime_list, \
                     "OptRelaxTime": CSB_optrelaxtime_list, "ImpRelaxTime": CSB_imprelaxtime_list}
                
                TCB_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": TCB_density_list, "Seebeck": TCB_seebeck_list, \
                     "Mobility": TCB_mobility_list, "Elcond": TCB_elcond_list, "Lorenz": TCB_lorenz_list, "Ekappa": TCB_ekappa_list, \
                     "Hallcoeff": TCB_hallcoeff_list, "PF": TCB_pf_list}
                
            if hasattr(self, 'VSB'):
                VSB_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": VSB_density_list, "Seebeck": VSB_seebeck_list, \
                     "Mobility": VSB_mobility_list, "Elcond": VSB_elcond_list, "Lorenz": VSB_lorenz_list, "Ekappa": VSB_ekappa_list, \
                     "Hallcoeff": VSB_hallcoeff_list, "PF": VSB_pf_list, "TotalRelaxTime": VSB_totalrelaxtime_list, "AcoRelaxTime": VSB_acorelaxtime_list, \
                     "OptRelaxTime": VSB_optrelaxtime_list, "ImpRelaxTime": VSB_imprelaxtime_list}
                
                TVB_dict={"Temperature": Temp_list, "dope": dope_list, "Concentration": TVB_density_list, "Seebeck": TVB_seebeck_list, \
                     "Mobility": TVB_mobility_list, "Elcond": TVB_elcond_list, "Lorenz": TVB_lorenz_list, "Ekappa": TVB_ekappa_list, \
                     "Hallcoeff": TVB_hallcoeff_list, "PF": TVB_pf_list}
        
        CBM_FILE = pd.DataFrame(CBM_dict)
        VBM_FILE = pd.DataFrame(VBM_dict)
        CBM_FILE.to_excel('CBM.xlsx', index_label='index', merge_cells=False)
        VBM_FILE.to_excel('VBM.xlsx', index_label='index', merge_cells=False)
        fp = open('Parameters', 'w')
        fp.write('band: m_parallel  m_vertical  m_cond  m_dos  E  N  Bandgap  c  epsilon_inf  epsilon_0 %s' % os.linesep)
        fp.write("CBM: %.4f %.4f %.4f %.4f %.4f %d %.4f %.4f %.4e %.4e %s" % (self.CBM.RT.ACO.EMC.parallelmass, self.CBM.RT.ACO.EMC.verticalmass,
                 self.CBM.RT.ACO.EMC.condeffmass, self.CBM.RT.ACO.doseffmass, self.CBM.RT.ACO.DPC.value, self.CBM.RT.ACO.N, self.CBM.RT.ACO.Bandgap,
                 self.CBM.RT.ACO.Elastic.value, self.CBM.RT.OPT.Diel.electron, self.CBM.RT.OPT.Diel.static, os.linesep))
        fp.write("VBM: %.4f %.4f %.4f %.4f %.4f %d %.4f %.4f %.4e %.4e %s" % (self.VBM.RT.ACO.EMC.parallelmass, self.VBM.RT.ACO.EMC.verticalmass,
                 self.VBM.RT.ACO.EMC.condeffmass, self.VBM.RT.ACO.doseffmass, self.VBM.RT.ACO.DPC.value, self.VBM.RT.ACO.N, self.VBM.RT.ACO.Bandgap,
                 self.VBM.RT.ACO.Elastic.value, self.VBM.RT.OPT.Diel.electron, self.VBM.RT.OPT.Diel.static, os.linesep))
        if hasattr(self, 'CSB'):
            CSB_FILE = pd.DataFrame(CSB_dict)
            TCB_FILE = pd.DataFrame(TCB_dict)
            CSB_FILE.to_excel('CSB.xlsx', index_label='index', merge_cells=False)
            TCB_FILE.to_excel('TCB.xlsx', index_label='index', merge_cells=False)
            fp.write("CSB: %.4f %.4f %.4f %.4f %.4f %d %.4f %.4f %.4e %.4e %s" % (self.CSB.RT.ACO.EMC.parallelmass, self.CSB.RT.ACO.EMC.verticalmass,
                     self.CSB.RT.ACO.EMC.condeffmass, self.CSB.RT.ACO.doseffmass, self.CSB.RT.ACO.DPC.value, self.CSB.RT.ACO.N, self.CSB.RT.ACO.Bandgap,
                     self.CSB.RT.ACO.Elastic.value, self.CSB.RT.OPT.Diel.electron, self.CSB.RT.OPT.Diel.static, os.linesep))
        if hasattr(self, 'VSB'):
            VSB_FILE = pd.DataFrame(VSB_dict)
            TVB_FILE = pd.DataFrame(TVB_dict)
            VSB_FILE.to_excel('VSB.xlsx', index_label='index', merge_cells=False)
            TVB_FILE.to_excel('TVB.xlsx', index_label='index', merge_cells=False)
            fp.write("VSB: %.4f %.4f %.4f %.4f %.4f %d %.4f %.4f %.4e %.4e %s" % (self.VSB.RT.ACO.EMC.parallelmass, self.VSB.RT.ACO.EMC.verticalmass,
                     self.VSB.RT.ACO.EMC.condeffmass, self.VSB.RT.ACO.doseffmass, self.VSB.RT.ACO.DPC.value, self.VSB.RT.ACO.N, self.VSB.RT.ACO.Bandgap,
                     self.VSB.RT.ACO.Elastic.value, self.VSB.RT.OPT.Diel.electron, self.VSB.RT.OPT.Diel.static, os.linesep))
        
        fp.close()        
        return
