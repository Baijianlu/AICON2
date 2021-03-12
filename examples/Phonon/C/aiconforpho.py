# -*- coding: utf-8 -*-
import numpy as np
from aicon.phonon import Get_Phonon

if __name__ == '__main__':
    Temp = np.arange(300, 1050, 50)
    ifscale = True
    Get_Phonon("./", Temp, ifscale)
