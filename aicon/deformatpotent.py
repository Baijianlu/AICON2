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
from pymatgen.io.vasp import BSVasprun
from pymatgen.electronic_structure.core import Spin

class DeformatPotentC(object):
    ''' Deformational potential constant class '''
    
    def __init__(self, flag):
        self.value = 0.0
        self.flag = flag
        self.englist = list()
        
        
    def __get__(self, obj, typ = None):
        return self.value
    
    def __str__(self):
        return '%.3f' % (self.value)
    
    __repr__ = __str__

    def Get_bandstruct(self, filepath):
        ''' Calculate band structure '''
        
        vaspband = BSVasprun(filepath + "vasprun.xml")
        bandstru = vaspband.get_band_structure(kpoints_filename=filepath+"KPOINTS",line_mode=True)
        return bandstru
           
    
    def DPCcalculator(self, filepath, *pos):
        ''' Get the energy of the band edge at different volume. '''
        
        bndindex = pos[0]
        kptindex = pos[1]
        for fi in filepath:
            bandstru = self.Get_bandstruct(fi)
            self.englist.append(bandstru.bands[Spin.up][bndindex,kptindex])
                
        
    def Get_Strain(self, filepath):
        ''' Get the strain. '''
        
        Vol = list()                                                             # number of strain = number of files - 1
        for fi in filepath:
            vasprun = BSVasprun(fi + "vasprun.xml")
            Vol.append(vasprun.final_structure.volume)
            
        self.strain = [np.abs(Vol[i]-Vol[0])/Vol[0] for i in np.arange(1, len(Vol))]
        
    
    def Get_DPC(self, filepath, *pos):
        ''' Calculate the deformaton potential constant. '''
        
        self.DPCcalculator(filepath, *pos)
        self.Get_Strain(filepath)
        DPC = [np.abs(self.englist[i] - self.englist[0])/(self.strain[i-1]) for i in np.arange(1,len(self.englist))]
        self.value = np.average(DPC)
         
        
        
