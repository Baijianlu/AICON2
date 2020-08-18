# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:40:58 2020

@author: Tao.Fan
"""
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
        self.seebeck = lambda x, T: Boltzm / C_e * quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]

    def Get_seebeck_P(self):
        ''' Calculate Seebeck coefficient with parabolic approximation. '''
        
        fun1 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (z - x) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 
        fun2 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 
        self.seebeck = lambda x, T: Boltzm / C_e * quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]        

    def Get_seebeck(self, ACO = True, ACO_P = False, OPT = False, OPT_P = False, IMP = False, IMP_P = False):
        ''' '''
        
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
        
        if self.flag == 'CSB' or self.flag == 'VSB':
            self.Get_seebeck_NP()
        else:
            self.Get_seebeck_NP()
        
               
