# -*- coding: utf-8 -*-
import numpy as np
from aicon.electron import Electron

if __name__ == '__main__':
    filepath = './equi'
    band = Electron()
    band.Get_bandstru(filepath)
    band.Get_SB()
    if hasattr(band,'CSB'):
        print('CSB: band index is %d, the coordinate is' % (band.CSB.pos['bndindex']+1))
        print(band.engband.kpoints[band.CSB.pos['kptindex']].frac_coords)
    if hasattr(band,'VSB'):
        print('VSB: band index is %d, the coordinate is' % (band.VSB.pos['bndindex']+1))
        print(band.engband.kpoints[band.VSB.pos['kptindex']].frac_coords)
