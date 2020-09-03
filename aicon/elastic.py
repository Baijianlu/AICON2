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
from numpy.linalg import inv
import pymatgen as pmg
from pymatgen.io.vasp import Outcar

class ElasticConst(object):
    ''' Elastic constant class. '''
    
    def __init__(self):
        self.value = 0.0
        self.tensor = list()
        
    def __get__(self, obj, typ = None):
        return self.value
    
    def __str__(self):
        return '%.3f' % (self.value)
        
    __repr__ = __str__
    
    def Get_ElasConst(self, filepath):
        outcar = Outcar(filepath + "OUTCAR")
        outcar.read_elastic_tensor()
        self.tensor = np.array(outcar.data["elastic_tensor"]) / 10       #unit in GPa
    
    def Get_AvgLongEConst(self, filepath):
        self.Get_ElasConst(filepath)
        self.value_2 = self.tensor[0,1] + 2*self.tensor[3,3] + 3/5 * (self.tensor[0,0] - self.tensor[0,1] - 2*self.tensor[3,3])
        
        self.Stensor = inv(self.tensor)
        struct = pmg.Structure.from_file(filepath + "POSCAR")
        self.density = struct.density * 1e3                              #kg/m^3
        B_v = ((self.tensor[0,0] + self.tensor[1,1] + self.tensor[2,2]) + 2 * (self.tensor[0,1] + self.tensor[1,2] + self.tensor[2,0])) / 9
        G_v = ((self.tensor[0,0] + self.tensor[1,1] + self.tensor[2,2]) - (self.tensor[0,1] + self.tensor[1,2] + self.tensor[2,0]) \
               + 3 * (self.tensor[3,3] + self.tensor[4,4] + self.tensor[5,5])) / 15
        B_r = 1 / ((self.Stensor[0,0] + self.Stensor[1,1] + self.Stensor[2,2]) + 2 * (self.Stensor[0,1] + self.Stensor[1,2] + self.Stensor[2,0]))
        G_r = 15 / (4*(self.Stensor[0,0] + self.Stensor[1,1] + self.Stensor[2,2]) - 4*(self.Stensor[0,1] + self.Stensor[1,2] + self.Stensor[2,0]) \
                    + 3 * (self.Stensor[3,3] + self.Stensor[4,4] + self.Stensor[5,5]))
        B_h = (B_v + B_r) / 2
        G_h = (G_v + G_r) / 2
        self.vel_long = ((B_h + 4/3 * G_h) * 1e9 / self.density)**(1/2)
        self.vel_tran = (G_h * 1e9 / self.density)**(1/2)
        self.vel_mean = 3 / (1/self.vel_long + 2/self.vel_tran)
        self.value = self.vel_mean**2 * self.density / 1e9
