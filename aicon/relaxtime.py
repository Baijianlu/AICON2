# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:04:38 2020

@author: Tao.Fan
"""
import numpy as np
import scipy.constants
from scipy.integrate import quad
from aicon.myemc import EffectMass
from aicon.deformatpotent import DeformatPotentC
from aicon.dielectric import DielConst
from aicon.elastic import ElasticConst

Planck = scipy.constants.hbar
EPlanck = scipy.constants.physical_constants['Planck constant over 2 pi in eV s'][0]
Boltzm = scipy.constants.Boltzmann
EBoltzm = scipy.constants.physical_constants['Boltzmann constant in eV/K'][0]
m_e = scipy.constants.m_e
EtoJoul = scipy.constants.physical_constants['joule-electron volt relationship'][0]
C_e = scipy.constants.e
epsilon_0 = scipy.constants.epsilon_0

def Upperlimit(x):
    if x < 0:
        return 100
    else:
        return x + 100
        
class RelaxTime(object):
    '''
    This is the parent class of all relaxation time classes with shared properties.    
    '''
    def __init__(self, flag, degeneracy, bandgap, *pos):
        self.value = 0.0
        self.flag = flag
        self.N = degeneracy
        self.Bandgap = bandgap
        self.Beta = lambda T: EBoltzm * T / self.Bandgap
        self.pos = pos
        
    def __get__(self, obj, typ = None):
        return self.value
    
    def __str__(self):
        return '%.2f' % self.value
        
    __repr__ = __str__   
    
    def Get_effemass(self, filepath):
        self.EMC = EffectMass()
        if self.flag == 'VBM':
            filepath = filepath + 'VBM/'
        elif self.flag == 'CBM':
            filepath = filepath + 'CBM/'
        elif self.flag == 'CSB':
            filepath = filepath + 'CSB/'
        else:
            filepath = filepath + 'VSB/'
                
        inpcar_fh = open(filepath+'INPCAR', 'r')
        kpt, stepsize, band, prg, basis = self.EMC.parse_inpcar(inpcar_fh)
        self.EMC.cal_effmass(kpt, stepsize, band, prg, basis, filepath+'EIGENVAL')
        self.effmass = lambda z, T: np.abs(self.EMC.condeffmass) * (1 + 2 * self.Beta(T) * z)
                        
    def Get_deformpot(self, filepath):
        self.DPC = DeformatPotentC(self.flag)            
        path = [filepath + 'equi/', filepath + '0.5per/', filepath + '1.0per/']                    
        self.DPC.Get_DPC(path, *self.pos)
    
    def Get_elastconst(self, filepath):
        self.Elastic = ElasticConst()
        self.Elastic.Get_AvgLongEConst(filepath)
                
    def Get_dielconst(self, filepath):
        self.Diel = DielConst()
        self.Diel.Get_DielConst(filepath)
                   
    def Set_degeneracy(self, value):
        self.N = value
    
    def Set_bandgap(self, value):
        self.Bandgap = value
        
    def Get_moment(self):
        self.Moment = lambda z, T: 1e-9 * np.sqrt(EtoJoul) * (2 * np.abs(self.EMC.doseffmass) * m_e)**(1/2) / EPlanck * (z * EBoltzm * T * (1 + z * EBoltzm * T / self.Bandgap))**(1/2)
        #unit is nm
    def Get_fermidistrFun(self, z, x):
        if np.exp(z-x) == np.inf:
            return 0.0
        else:
            return 1.0 / (np.exp(z - x) + 1)
    
    def DfermidistrFun(self, z, x):
        if np.exp(z-x)**2 == np.inf:
            return 0.0
        else:
            return -np.exp(z - x) / (np.exp(z - x) + 1)**2
        
    def integral(self, x, T, n, m, k):
        integrand = lambda z, x, T: (-self.DfermidistrFun(z,x)) * z**(n) * (z + self.Beta(T) * z**2)**(m) * (1 + 2 * self.Beta(T) * z)**(k)        
        return quad(integrand, 0, Upperlimit(x), args=(x, T))[0]
    
    def Fermiintegral(self, x, n):
        integrand = lambda z, x: self.Get_fermidistrFun(z, x) * z**(n)
        return quad(integrand, 0, np.inf, args=(x))[0]
        
        
class AcoRelaxTime(RelaxTime):
    ''' this is the subclass of RelaxTime, to represent acoustic phonon relaxation time with non-parabolic approximation'''
    
    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2) \
        * (1 + 2 * z * EBoltzm * T / self.Bandgap) * (1 + z * EBoltzm * T / self.Bandgap)**(1/2)                                     #here I omit the constant EtoJoul^3/2
    
    def Get_relaxtimefun(self, filepath):
        self.Get_deformpot(filepath)              
        self.Get_elastconst(filepath + 'elastic/')
        self.Get_effemass(filepath)
        self.Beta = lambda T: EBoltzm * T / self.Bandgap
        self.Get_DOSfun()
        self.Acotime = lambda z, T: 1 / (np.sqrt(EtoJoul) * np.pi * EBoltzm * T * self.DOS(z, T) * self.DPC.value**2 / (EPlanck * self.Elastic.value * 1e9 * self.N) \
                                         * (1 - 8 * self.Beta(T) * (z + self.Beta(T) * z**2) / (3 * (1 + 2 * self.Beta(T) * z)**2)))
        
    def Get_Avgacotime(self, filepath):
        self.Get_relaxtimefun(filepath)
        self.Get_moment()
        fun1 = lambda z, x, T: 1e14 * self.Acotime(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        fun4 = lambda z, x, T: 1e14 * self.Acotime(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3 / (1 + 2 * self.Beta(T) * z)
        fun5 = lambda z, x, T: self.effmass(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3 
        self.Avgacotime2 = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * 1e-14
        self.Avgacotime = lambda x, T: quad(fun4, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * 1e-14
        self.Avgeffmass = lambda x, T: quad(fun5, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]


class AcoRelaxTime_Para(RelaxTime):
    ''' this is the subclass of RelaxTime, to represent acoustic phonon relaxation time with parabolic approximation'''
    
    def Get_moment(self):
        self.Moment = lambda z, T: 1e-9 * np.sqrt(EtoJoul) * (2 * np.abs(self.EMC.doseffmass) * m_e)**(1/2) / EPlanck * (z * EBoltzm * T)**(1/2)    
    
    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2)
        
    def Get_relaxtimefun(self, filepath):
        self.Get_deformpot(filepath)
        self.Get_elastconst(filepath + 'elastic/')
        self.Get_effemass(filepath)
        self.Get_DOSfun()
        self.Acotime = lambda z, T: 1 / (np.sqrt(EtoJoul) * np.pi * EBoltzm * T * self.DOS(z, T) * self.DPC.value**2 / (EPlanck * self.Elastic.value * 1e9 * self.N))

    def Get_Avgacotime(self, filepath):
        self.Get_relaxtimefun(filepath)
        self.Get_moment()
        fun1 = lambda z, x, T: 1e14 * self.Acotime(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        self.Avgacotime = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * 1e-14
        self.Avgeffmass = lambda x, T: np.abs(self.EMC.condeffmass)


class OptRelaxTime(RelaxTime):    
    '''this is the subclass of RelaxTime, to represent polar optical phonon relaxation time with non-parabolic approximation'''
    
    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2) \
        * (1 + 2 * z * self.Beta(T)) * (1 + z * self.Beta(T))**(1/2)
        
    def Get_scrad(self):
        self.Get_DOSfun()  
        self.Scrad = lambda z, T: EtoJoul**(5/2) * 4 * np.pi * C_e**2 * self.DOS(z, T) / self.Diel.electron
           
    def Get_delta(self, filepath):
        self.Get_effemass(filepath)
        self.Get_dielconst(filepath + "dielect/")
        self.Get_scrad()
        self.Get_moment()
        self.Delta = lambda z, T: (2 * self.Moment(z, T) * 1e9)**(-2) * self.Scrad(z, T)
        
    def Get_relaxtimefun(self, filepath):
        self.Get_delta(filepath)      
        self.Opttime = lambda z, T: 1/(EtoJoul**(3/2) * 2**(1/2) * EBoltzm * T * C_e**2 * np.abs(self.EMC.doseffmass * m_e)**(1/2) * (self.Diel.electron**(-1) - self.Diel.static**(-1)) / (EPlanck**2 * (z * EBoltzm * T)**(1/2)) \
                                         * (1 + 2 * self.Beta(T) * z) / (1 + self.Beta(T) * z)**(1/2) * ((1 - self.Delta(z, T) * np.log(1 + 1 / self.Delta(z, T))) - 2 * self.Beta(T) * (z + self.Beta(T) * z**2) \
                                            / (1 + 2 * self.Beta(T) * z)**2 * (1 - 2 * self.Delta(z, T) + 2 * self.Delta(z, T)**2 * np.log(1 + 1 / self.Delta(z, T)))))
        
    def Get_Avgopttime(self, filepath):
        self.Get_relaxtimefun(filepath)
        fun1 = lambda z, x, T: 1e14 * self.Opttime(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3 
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        self.Avgopttime = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * 1e-14
        

class OptRelaxTime_Para(RelaxTime):
    '''this is the subclass of RelaxTime, to represent polar optical phonon relaxation time with parabolic approximation'''
    
    def Get_moment(self):
        self.Moment = lambda z, T: 1e-9 * np.sqrt(EtoJoul) * (2 * np.abs(self.EMC.doseffmass) * m_e)**(1/2) / EPlanck * (z * EBoltzm * T)**(1/2) 

    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2)

    def Get_scrad(self, filepath):  
        self.Get_DOSfun()             
        self.Scrad = lambda z, T: EtoJoul**(5/2) * 4 * np.pi * C_e**2 * self.DOS(z, T) / self.Diel.electron
        
    def Get_delta(self, filepath):
        self.Get_effemass(filepath)
        self.Get_dielconst(filepath + "dielect/") 
        self.Get_scrad(filepath)
        self.Get_moment()
        self.Delta = lambda z, T: (2 * self.Moment(z, T) * 1e9)**(-2) * self.Scrad(z, T)
        
    def Get_relaxtimefun(self, filepath):
        self.Get_delta(filepath)
        self.Opttime = lambda z, T: 1/(EtoJoul**(3/2) * 2**(1/2) * EBoltzm * T * C_e**2 * np.abs(self.EMC.doseffmass * m_e)**(1/2) * (self.Diel.electron**(-1) - self.Diel.static**(-1)) / (EPlanck**2 * (z * EBoltzm * T)**(1/2)) \
                                         * (1 - self.Delta(z, T) * np.log(1 + 1 / self.Delta(z, T))))
        
    def Get_Avgopttime(self, filepath):
        self.Get_relaxtimefun(filepath)
        fun1 = lambda z, x, T: 1e14 * self.Opttime(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3 
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        self.Avgopttime = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0] * 1e-14

class ImpurityRelaxTime(RelaxTime):
    '''this is the subclass of RelaxTime, to represent ionic impurity relaxation time with non-parabolic approximation'''
    
    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2) \
        * (1 + 2 * z * self.Beta(T)) * (1 + z * self.Beta(T))**(1/2)                                                                                    #here I omit the constant EtoJoul^3/2
        self.Density = lambda x, T: EtoJoul**(3/2) * (2 * np.abs(self.doseffmass) * m_e * EBoltzm * T)**(3/2) / (3 * np.pi**2 * EPlanck**3) \
        * self.integral(x, T, 0, 3/2, 0)     
    
    def Get_AvgDOS(self, filepath):
        self.Get_effemass(filepath)
        self.Get_DOSfun()
        self.Get_moment()
        fun1 = lambda z, x, T: self.DOS(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        self.AvgDOS = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]
        
    def Get_scrad(self, filepath):  
        self.Get_AvgDOS(filepath)
        self.Get_dielconst(filepath + "dielect/")              
        self.Scrad_2 = lambda x, T: EtoJoul**(5/2) * 4 * np.pi * C_e**2 * self.AvgDOS(x, T) / self.Diel.static
    
    def Get_kF(self):
        self.k_F = lambda x, T: (3 * np.pi**2 * self.Density(x, T) / self.N)**(1/3)
        
    def Get_delta(self, filepath):
        self.Get_scrad(filepath)
        self.Get_kF()
        self.Delta_2 = lambda x, T: (2 * self.k_F(x, T))**(-2) * self.Scrad_2(x, T)

    def Get_Imptime(self, filepath):
        self.Get_delta(filepath)
        self.Imptime_2 = lambda x, T: 1 / (EtoJoul**3 * 2 * C_e**4 * self.N * np.abs(self.EMC.doseffmass) * m_e * (1 + 2 * x * self.Beta(T)) \
                                         * (np.log(1 + 1/self.Delta_2(x, T)) - (1 + self.Delta_2(x, T))**(-1)) \
                                         / (3 * np.pi * self.Diel.static**2 * EPlanck**3))
        
        self.Avgimptime = self.Imptime_2
        
class ImpurityRelaxTime_Para(RelaxTime):
    '''this is the subclass of RelaxTime, to represent ionic impurity relaxation time with parabolic approximation'''
    
    def Get_moment(self):
        self.Moment = lambda z, T: 1e-9 * np.sqrt(EtoJoul) * (2 * np.abs(self.EMC.doseffmass) * m_e)**(1/2) / EPlanck * (z * EBoltzm * T)**(1/2)                        #unit nm
        
    def Get_DOSfun(self):
        self.doseffmass = self.N**(2/3) * self.EMC.doseffmass 
        self.DOS = lambda z, T: 2**(1/2) * np.abs(self.doseffmass * m_e)**(3/2) / (np.pi**2 * EPlanck**3) * (z * EBoltzm * T)**(1/2)                                    #here I omit the constant EtoJoul^3/2
        self.Density = lambda x, T: EtoJoul**(3/2) * 2**(1/2) * (np.abs(self.doseffmass) * m_e * EBoltzm * T)**(3/2) / (np.pi**2 * EPlanck**3) \
        * self.Fermiintegral(x, 1/2)     
    
    def Get_AvgDOS(self, filepath):
        self.Get_effemass(filepath)
        self.Get_DOSfun()
        self.Get_moment()
        fun1 = lambda z, x, T: self.DOS(z, T) * (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        fun2 = lambda z, x, T: (-self.DfermidistrFun(z,x)) * self.Moment(z, T)**3
        self.AvgDOS = lambda x, T: quad(fun1, 0, Upperlimit(x), args=(x, T))[0] / quad(fun2, 0, Upperlimit(x), args=(x, T))[0]
            
    def Get_scrad(self, filepath):  
        self.Get_AvgDOS(filepath)
        self.Get_dielconst(filepath + "dielect/")              
        self.Scrad_2 = lambda x, T: EtoJoul**(5/2) * 4 * np.pi * C_e**2 * self.AvgDOS(x, T) / self.Diel.static
    
    def Get_kF(self):
        self.k_F = lambda x, T: (3 * np.pi**2 * self.Density(x, T) / self.N)**(1/3)
        
    def Get_delta(self, filepath):
        self.Get_scrad(filepath)
        self.Get_kF()
        self.Delta_2 = lambda x, T: (2 * self.k_F(x, T))**(-2) * self.Scrad_2(x, T)
        
    def Get_Imptime(self, filepath):
        self.Get_delta(filepath)
        self.Imptime_2 = lambda x, T: 1 / (EtoJoul**3 * 2 * C_e**4 * self.N * np.abs(self.EMC.doseffmass) * m_e * (1 + 2 * x * self.Beta(T)) \
                                         * (np.log(1 + 1/self.Delta_2(x, T)) - (1 + self.Delta_2(x, T))**(-1)) \
                                         / (3 * np.pi * self.Diel.static**2 * EPlanck**3))
 
        self.Avgimptime = self.Imptime_2
        
class TotalRelaxTime(RelaxTime):
    '''this is the subclass of RelaxTime, to represent total relaxation time'''
    
    def Get_Totaltime(self, filepath, ACO = True, ACO_P = False, OPT = False, OPT_P = False, IMP = False, IMP_P = False):
        fun1 = lambda x, T: 0
        fun2 = lambda x, T: 0
        fun3 = lambda x, T: 0
        
        if ACO == True:
            self.ACO = AcoRelaxTime(self.flag, self.N, self.Bandgap, *self.pos)
            self.ACO.Get_Avgacotime(filepath)
            fun1 = lambda x, T: 1/self.ACO.Avgacotime(x, T)
        if ACO_P == True:
            self.ACO = AcoRelaxTime_Para(self.flag, self.N, self.Bandgap, *self.pos)
            self.ACO.Get_Avgacotime(filepath)
            fun1 = lambda x, T: 1/self.ACO.Avgacotime(x, T)
        if OPT == True:
            self.OPT = OptRelaxTime(self.flag, self.N, self.Bandgap)
            self.OPT.Get_Avgopttime(filepath)
            if self.OPT.Diel.ion == 0:
                pass
            else:
                fun2 = lambda x, T: 1/self.OPT.Avgopttime(x, T)
        if OPT_P == True:
            self.OPT = OptRelaxTime_Para(self.flag, self.N, self.Bandgap)
            self.OPT.Get_Avgopttime(filepath)
            if self.OPT.Diel.ion == 0:
                pass
            else:
                fun2 = lambda x, T: 1/self.OPT.Avgopttime(x, T)
        if IMP == True:
            self.IMP = ImpurityRelaxTime(self.flag, self.N, self.Bandgap)
            self.IMP.Get_Imptime(filepath)
            fun3 = lambda x, T: 1/self.IMP.Avgimptime(x, T)
        if IMP_P == True:
            self.IMP = ImpurityRelaxTime_Para(self.flag, self.N, self.Bandgap)
            self.IMP.Get_Imptime(filepath)
            fun3 = lambda x, T: 1/self.IMP.Avgimptime(x, T)
            
        self.SR = lambda x, T: fun1(x, T) + fun2(x, T) + fun3(x, T)
        self.Totaltime = lambda x, T: 1/self.SR(x, T)
        self.doseffmass = self.ACO.doseffmass
        self.Avgeffmass = self.ACO.Avgeffmass
        
    

