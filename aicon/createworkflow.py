# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:20:45 2020

@author: Tao.Fan
"""
import numpy as np
from pymatgen import Structure
from myprocesscontrol.myworkflow import wf_electron_conductivity
from fireworks import LaunchPad
from myprocesscontrol.tools import Generate_kpoints

if __name__ == '__main__':
    struct = Structure.from_file("POSCAR")
    custom_settings_relax = {"SYSTEM": "PbTe",
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
    custom_settings_static = {"SYSTEM": "PbTe",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "EDIFF": 1E-7,
    "EDIFFG": -1E-3,
    "PREC": "Accurate",
    "NPAR": 2}
    custom_settings_band = {"SYSTEM": "PbTe",
    "ENCUT": 600,
    "ISTART": 1,
    "ICHARG": 11,
    "ISMEAR": 0,
    "SIGMA": 0.05,
    "LPLANE": ".True.",
    "LREAL": ".FALSE.",
    "NBANDS": 28,
    "LMAXMIX": 4,
    "ADDGRID": ".TRUE.",
    "ALGO": "A",
    "EDIFF": 1E-7,
    "PREC": "Accurate",
    "NPAR": 4}    
    custom_settings_diel = {"SYSTEM": "PbTe",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "NSW": 1,
    "IBRION": 8,
    "EDIFF": 1E-7,
    "EDIFFG": -1E-3,
    "PREC": "Accurate",
    "LEPSILON": ".TRUE.",
    "LRPA": ".FALSE."}
    custom_settings_elastic = {"SYSTEM": "PbTe",
    "ENCUT": 600,
    "ISTART": 0,
    "ICHARG": 2,
    "ISMEAR": -5,
    "IBRION": 6,
    "NFREE": 4,
    "ISIF": 3,
    "POTIM": 0.01,
    "EDIFF": 1E-7,
    "EDIFFG": -1E-3,
    "PREC": "Accurate",
    "ADDGRID": ".TRUE.",
    "LWAVE": ".FALSE."}
    
    KPOINT = Generate_kpoints(struct, 0.03)
    
    # set up the LaunchPad and reset it
    launchpad = LaunchPad(host="mongodb+srv://User1:3327580@test-2a9ni.gcp.mongodb.net/test?retryWrites=true&w=majority", name="VASP")
    launchpad.reset('', require_password=False)
    
    workflow = wf_electron_conductivity(struct, vasp_input_set_relax=custom_settings_relax, vasp_input_set_static=custom_settings_static, vasp_input_set_band=custom_settings_band, vasp_input_set_diel=custom_settings_diel, vasp_input_set_elastic=custom_settings_elastic, vasp_kpoint_set=KPOINT, vasp_cmd=">>vasp_cmd<<", db_file=">>db_file<<")
    
    # store workflow and launch it locally
    launchpad.add_wf(workflow)
