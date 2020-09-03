# -*- coding: utf-8 -*-
import numpy as np
#from myprocesscontrol.aicon import Get_Electron
from aicon.electron import Electron

if __name__ == '__main__':
    mode = 'standard'
    Temp = np.arange(200, 350, 50)
    dope = [1.1e+19, 3.3e+19, 5e+19]
    ifSB = True
#    Get_Electron("./", Temp, Doping, mode, ifSB)
    Compound = Electron()
    Compound.Get_bandstru("./equi/")
    Compound.Get_values("./", Temp, dope, mode, ifSB=ifSB)
    Compound.Output(Temp, dope, mode)
