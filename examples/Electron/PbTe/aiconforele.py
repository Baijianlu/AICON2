# -*- coding: utf-8 -*-
import numpy as np
from aicon.electron import Get_Electron

if __name__ == '__main__':
    mode = 'doping'
    Temp = np.arange(300, 950, 50)
    dope = [1.8e+19, 2.5e+20]
    ifSB = True
    Get_Electron("./", Temp, dope, mode, ifSB)
