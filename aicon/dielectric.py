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
from pymatgen.io.vasp import Vasprun
import scipy.constants
epsilon_0 = scipy.constants.epsilon_0

class DielConst(object):
    ''' Dielectric constant class'''
    
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
        
        
