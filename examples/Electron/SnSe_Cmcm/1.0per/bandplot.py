# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:33:43 2019

@author: nwpuf
"""

from pymatgen.io.vasp import Vasprun, BSVasprun
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
import matplotlib.pyplot as plt

bsv = BSVasprun("vasprun.xml")
bs = bsv.get_band_structure(kpoints_filename="KPOINTS",line_mode=True)
print(bs.get_band_gap())
bsplot = BSPlotter(bs)
bsplot.get_plot(vbm_cbm_marker=True).show()

#dosrun = Vasprun("DOS/vasprun.xml", parse_dos=True)
#dos = dosrun.complete_dos
#dosplot = DosPlotter(sigma=0.1)
#dosplot.add_dos("Total DOS", dos)
#dosplot.add_dos_dict(dos.get_element_dos())
#ax = plt.gca()
#print(type(dosplot.get_plot()))
#dosplot.get_plot().show()

