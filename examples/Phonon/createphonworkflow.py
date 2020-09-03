# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:20:45 2020

@author: Tao.Fan
"""
import numpy as np
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from aicon.myworkflow import wf_phonon_conductivity
from fireworks import LaunchPad
from aicon.tools import Generate_kpoints

if __name__ == '__main__':
    struct = Structure.from_file("POSCAR")
    finder = SpacegroupAnalyzer(struct)
    struct=finder.get_conventional_standard_structure()
    custom_settings_relax = {"SYSTEM": "Si",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "SIGMA": 0.02,
    "NSW": 60,
    "IBRION": 2,
    "ISIF": 3,
    "POTIM": 0.05,
    "EDIFF": 1E-7,
    "EDIFFG": -1E-3,
    "PREC": "Accurate",
    "NPAR": 2}
    custom_settings_fixvol_relax = {"SYSTEM": "Si",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "SIGMA": 0.02,
    "NSW": 60,
    "IBRION": 2,
    "ISIF": 4,
    "POTIM": 0.05,
    "EDIFF": 1E-7,
    "EDIFFG": -1E-3,
    "PREC": "Accurate",
    "NPAR": 2}
    custom_settings_dfpt = {"SYSTEM": "Si",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "SIGMA": 0.02,
    "IBRION": 8,
    "IALGO": 38,
    "EDIFF": 1E-7,
    "EDIFFG": 1E-6,
    "PREC": "Accurate",
    "LCHARG": ".FALSE.",
    "LWAVE": ".FALSE.",
    "LREAL": ".FALSE.",
    "ADDGRID": ".TRUE."}
    
    KPOINT = Generate_kpoints(struct, 0.03)
    
    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host=<your database url>, name="test")
    launchpad.reset('', require_password=False)
    
    workflow = wf_phonon_conductivity(struct, vasp_input_set_relax=custom_settings_relax, vasp_input_set_fixvol_relax=custom_settings_fixvol_relax, vasp_input_set_dfpt=custom_settings_dfpt, vasp_kpoint_set=KPOINT, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<", Temp=[300], supercell=[2,2,2])
    
    # store workflow and launch it locally
    launchpad.add_wf(workflow)
