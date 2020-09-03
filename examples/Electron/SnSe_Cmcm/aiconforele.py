# -*- coding: utf-8 -*-
import numpy as np
from aicon.electron import Get_Electron

if __name__ == '__main__':
    mode = 'doping'
    Temp = np.arange(300, 950, 50)
    Doping = [5e+18, 9e+18]
    ifSB = False
    Get_Electron("./", Temp, Doping, mode, ifSB)
