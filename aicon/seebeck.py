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

import scipy.constants
from scipy.integrate import quad

Boltzm = scipy.constants.Boltzmann
C_e = scipy.constants.e

def Upperlimit(x):
    if x < 0:
        return 100
    else:
        return x + 100

class Seebeck(object):
    '''Seebeck coefficient class'''
    
    def __init__(self, flag, RelaxTime):
        self.value = 0.0
        self.flag = flag
        self.RelaxT = RelaxTime
        
    def Get_seebeck_NP(self):
        ''' Calculate Seebeck coefficient with non-parabolic approximation. '''
        
        fun1 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (z - x) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 / (1 + 2 * self.RelaxT.Beta(T) * z)
        fun2 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 / (1 + 2 * self.RelaxT.Beta(T) * z)
        self.integfun1 = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0]
        self.integfun2 = lambda x, T: quad(fun2, 0, Upperlimit(x), args=(x, T))[0]

    def Get_seebeck_P(self):
        ''' Calculate Seebeck coefficient with parabolic approximation. '''
        
        fun1 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (z - x) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 
        fun2 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 
        self.seebeck = lambda x, T: Boltzm / C_e * quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]        

    def Get_seebeck(self, ACO = True, ACO_P = False, OPT = False, OPT_P = False, IMP = False, IMP_P = False):
        '''
        Calculate Seebeck coefficient.
        
        Parameters:
        ----------
        ACO: bool
            If consider the acoustic phonon scattering in total relaxation time.
        ACO_P: bool
            If consider the acoustic phonon scattering with parabolic approximation in total relaxation time.
        '''
        
        fun1 = lambda z, T: 0
        fun2 = lambda z, T: 0
        fun3 = lambda z, T: 0
        
        if ACO == True or ACO_P == True:
            fun1 = lambda z, T: 1/self.RelaxT.ACO.Acotime(z, T)

        if OPT == True or OPT_P == True:
            if self.RelaxT.OPT.Diel.ion == 0:
                pass
            else:
                fun2 = lambda z, T: 1/self.RelaxT.OPT.Opttime(z, T)
        
        if IMP == True or IMP_P == True:
            fun3 = lambda z, T: 1/self.RelaxT.IMP.Imptime(z, T)
            
        self.Totaltime = lambda z, T: 1 / (fun1(z, T) + fun2(z, T) + fun3(z, T))
        
        self.Get_seebeck_NP()
        
               
