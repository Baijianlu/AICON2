# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 16:36:10 2019

@author: Tao.Fan
"""
import numpy as np
from pymatgen.io.vasp import BSVasprun
from pymatgen.electronic_structure.core import Spin

class DeformatPotentC(object):
    '''This class is used to calculate deformational potential constant and store the data  '''
    
    ###class method######################################################################
    def __init__(self, flag):
        self.value = 0.0
        self.flag = flag
        self.englist = list()
        
        
    def __get__(self, obj, typ = None):
        return self.value
    
    def __str__(self):
        return '%.3f' % (self.value)
    
    __repr__ = __str__

    def Get_bandstruct(self, filepath):                    #CBM,VBM SB could be merged
        vaspband = BSVasprun(filepath + "vasprun.xml")
        bandstru = vaspband.get_band_structure(kpoints_filename=filepath+"KPOINTS",line_mode=True)
        return bandstru
           
    
    def DPCcalculator(self, filepath, *pos):
        bndindex = pos[0]
        kptindex = pos[1]
        for fi in filepath:
            bandstru = self.Get_bandstruct(fi)
            self.englist.append(bandstru.bands[Spin.up][bndindex,kptindex])
                
        
    def Get_Strain(self, filepath):
        Vol = list()                                         # number of strain = number of files - 1
        for fi in filepath:
            vasprun = BSVasprun(fi + "vasprun.xml")
            Vol.append(vasprun.final_structure.volume)
            
        self.strain = [np.abs(Vol[i]-Vol[0])/Vol[0] for i in np.arange(1, len(Vol))]
        return
    
    def Get_DPC(self, filepath, *pos):        
        self.DPCcalculator(filepath, *pos)
        self.Get_Strain(filepath)
        DPC = [np.abs(self.englist[i] - self.englist[0])/(self.strain[i-1]) for i in np.arange(1,len(self.englist))]
        self.value = np.average(DPC)
        return DPC 
        
        
