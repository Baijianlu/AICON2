# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:48:52 2020

@author: Tao.Fan
"""
import numpy as np
from pymatgen.io.vasp import Vasprun
import scipy.constants
epsilon_0 = scipy.constants.epsilon_0

class DielConst(object):
    ''' this class is used to calculate dielectric constant and store the data'''
    
    def __init__(self):
        self.electron = 0.0
        self.ion = 0.0
        
    def __get__(self, obj, typ = None):
        return self.electron, self.ion
    
    def __str__(self):
        return '%.2f %.2f' % (self.electron, self.ion)
        
    __repr__ = __str__
    
    def Get_DielTensor(self, filepath):
        vasprun = Vasprun(filepath + "vasprun.xml")
        self.electensor = np.array(vasprun.epsilon_static)
        self.iontensor = np.array(vasprun.epsilon_ionic)
        
    def Get_DielConst(self, filepath):
        self.Get_DielTensor(filepath)
        self.electron = np.average(np.linalg.eig(self.electensor)[0]) * epsilon_0
        self.ion = np.average(np.linalg.eig(self.iontensor)[0]) * epsilon_0
        self.static = self.electron + self.ion
        return self.electron, self.ion
        
