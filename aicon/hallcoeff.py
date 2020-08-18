# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:25:58 2020

@author: Tao.Fan
"""
import scipy.constants
from scipy.integrate import quad

c = scipy.constants.c
C_e = scipy.constants.e

def Upperlimit(x):
    if x < 0:
        return 100
    else:
        return x + 100

class Hallcoeff(object):
    '''Hall coefficient class'''
    
    def __init__(self, flag, RelaxTime):
        self.value = 0.0
        self.flag = flag
        self.RelaxT = RelaxTime
        
    def Get_AK(self):
        ''' Calculate effective mass anisotropy factor. '''
        
        K = self.RelaxT.ACO.EMC.parallelmass / self.RelaxT.ACO.EMC.verticalmass
        self.A_K = 3 * K * (K + 2) / (2 * K + 1)**2
        
    def Get_A_NP(self):
        ''' Calculate Hall factor with non-parabolic approximation.'''
        
        fun1 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 / (1 + 2 * self.RelaxT.Beta(T) * z)
        fun2 = lambda z, x, T: (1e14 * self.Totaltime(z, T))**2 * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3 / (1 + 2 * self.RelaxT.Beta(T) * z)**2
        fun3 = lambda z, x, T: (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3
        self.A = lambda x, T: self.A_K * quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * quad(fun3, 0, Upperlimit(x), args=(x, T))[0] / (quad(fun1, 0, Upperlimit(x), args=(x, T))[0])**2

    def Get_A_P(self):
        ''' Calculate Hall factor with parabolic approximation. '''
        
        fun1 = lambda z, x, T: 1e14 * self.Totaltime(z, T) * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3
        fun2 = lambda z, x, T: (1e14 * self.Totaltime(z, T))**2 * (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3
        fun3 = lambda z, x, T: (-self.RelaxT.DfermidistrFun(z,x)) * self.RelaxT.ACO.Moment(z, T)**3
        self.A = lambda x, T: self.A_K * quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * quad(fun3, 0, Upperlimit(x), args=(x, T))[0] / (quad(fun1, 0, Upperlimit(x), args=(x, T))[0])**2

    def Get_hallcoeff(self, density, ACO = True, ACO_P = False, OPT = False, OPT_P = False, IMP = False, IMP_P = False):
        ''' 
        Calculate Hall coefficient. 
        
        Parameters:
        ----------
        density: function
            The carrier concentration function.
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
        
        self.Get_AK()
        
        if self.flag == 'CSB' or self.flag == 'VSB':
            self.Get_A_NP()
        else:
            self.Get_A_NP()

        self.hallcoeff = lambda x, T: self.A(x, T) / (C_e * density(x, T))
        
        
