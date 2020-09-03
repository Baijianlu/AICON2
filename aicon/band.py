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

import numpy as np
import scipy.constants
from pymatgen.io.vasp import BSVasprun
from aicon.relaxtime import AcoRelaxTime, AcoRelaxTime_Para, ImpurityRelaxTime, TotalRelaxTime
from aicon.seebeck import Seebeck
from aicon.ekappa import Ekappa
from aicon.hallcoeff import Hallcoeff

m_e = scipy.constants.m_e
C_e = scipy.constants.e
EPlanck = scipy.constants.physical_constants['Planck constant over 2 pi in eV s'][0]
EBoltzm = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
EtoJoul = scipy.constants.physical_constants['joule-electron volt relationship'][0]

class Band(object):
    '''This class represents one band edge considered in transport property calculation'''
    
    def __init__(self, gap, degeneracy, isCBM = False, isVBM = False, isCSB = False, isVSB = False, **pos):
        '''
        Parameters:
        ----------
        gap: float
            band gap, the unit is eV.
        degeneracy: int
            the band multiplicity due to the symmetry.
        \*\*pos: dict
            The position of band edge, including kpoint index and band number index.
        
        '''
        self.value = 0.0
        if isCBM is True:
            self.flag = "CBM"
        elif isVBM is True:
            self.flag = "VBM"
        elif isCSB is True:
            self.flag = "CSB"
        elif isVSB is True:
            self.flag = "VSB" 
        else:
            pass                                                                                #here should raise a warning
        
        self.bandgap = gap
        self.degeneracy = degeneracy
        self.pos = pos
        
    def __get__(self, obj, typ = None):
        return self.value
    
    def __str__(self):
        return '%.2f' % self.value
        
    __repr__ = __str__
    
    def Get_gap_degeneracy(self, filepath):
        vaspband = BSVasprun(filepath + "/vasprun.xml")
        bandstru = vaspband.get_band_structure(kpoints_filename=filepath+"/KPOINTS",line_mode=True)
        self.bandgap = bandstru.get_band_gap()['energy']
        self.degeneracy = bandstru.get_kpoint_degeneracy(bandstru.kpoints[self.pos['kptindex']].frac_coords)
        
        
    def Get_relaxtime(self, filepath):
        if self.flag == 'CSB' or self.flag == 'VSB':            
            self.RT = TotalRelaxTime(self.flag, self.degeneracy, self.bandgap, self.pos['bndindex'], self.pos['kptindex'])
            self.RT.Get_Totaltime(filepath, ACO = True, OPT = True, IMP = True)
        else:
            self.RT = TotalRelaxTime(self.flag, self.degeneracy, self.bandgap, self.pos['bndindex'], self.pos['kptindex'])
            self.RT.Get_Totaltime(filepath, ACO = True, OPT = True, IMP = True)                                                      #unit: second
                
    def Get_mobility(self, filepath):
        if not hasattr(self, 'RT'):        
            self.Get_relaxtime(filepath)
            
        self.Mobility = lambda x, T: C_e * self.RT.Totaltime(x, T) / (self.RT.Avgeffmass(x,T) * m_e)           #unit: m^2 * V^-1 * S^-1

    def Get_carridensity(self, filepath):
        if not hasattr(self, 'RT'):        
            self.Get_relaxtime(filepath)
        
        if self.flag == 'CSB' or self.flag == 'VSB':
            self.Density = lambda x, T: EtoJoul**(3/2) * (2 * np.abs(self.RT.doseffmass) * m_e * EBoltzm * T)**(3/2) / (3 * np.pi**2 * EPlanck**3) \
            * self.RT.integral(x, T, 0, 3/2, 0)
        else:
            self.Density = lambda x, T: EtoJoul**(3/2) * (2 * np.abs(self.RT.doseffmass) * m_e * EBoltzm * T)**(3/2) / (3 * np.pi**2 * EPlanck**3) \
            * self.RT.integral(x, T, 0, 3/2, 0)                                                               #unit: m^-3

    
    def Get_eleconduct(self, filepath):
        if not hasattr(self, 'Mobility'):
            self.Get_mobility(filepath)
        if not hasattr(self, 'Density'):
            self.Get_carridensity(filepath)
        
        self.Elcond = lambda x, T: C_e * self.Density(x, T) * self.Mobility(x, T)                             #unit: 1/m*ohm
        
    def Get_seebeck(self, filepath):
        if not hasattr(self, 'RT'):        
            self.Get_relaxtime(filepath)
            
        self.Seebeck = Seebeck(self.flag, self.RT)
        self.Seebeck.Get_seebeck(ACO = True, OPT = True, IMP = False)                                        #unit: V/K
        
    def Get_ekappa(self, filepath):
        if not hasattr(self, 'RT'):        
            self.Get_relaxtime(filepath)
            
        self.Ekappa = Ekappa(self.flag, self.RT)
        self.Ekappa.Get_ekappa(self.Elcond, ACO = True, OPT = True, IMP = False)                            #unit: W/m*K
                
    def Get_hallcoeff(self, filepath):
        if not hasattr(self, 'RT'):        
            self.Get_relaxtime(filepath)
            
        self.Hallcoeff = Hallcoeff(self.flag, self.RT)
        self.Hallcoeff.Get_hallcoeff(self.Density, ACO = True, OPT = True, IMP = False)                     #unit: m^3/C
            
            
            
            
            
            
            
            
            
