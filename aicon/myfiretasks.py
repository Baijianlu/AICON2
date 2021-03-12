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

import os
import numpy as np
from fireworks import explicit_serialize, FiretaskBase, FWAction
from pymatgen import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Oszicar,VaspParserError
from pymatgen.io.vasp.sets import MPStaticSet, MPNonSCFSet, MPRelaxSet
from pymatgen.electronic_structure.core import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from atomate.common.firetasks.glue_tasks import get_calc_loc, CopyFiles
from atomate.utils.utils import env_chk
from aicon.tools import Generate_kpoints, Write_INPCAR, get_highsympath
from aicon.electron import Electron, Get_Electron
from aicon.myemc import EffectMass
from aicon.phonon import Get_Phonon
import phonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml
from phonopy.interface.calculator import get_default_physical_units
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.interface.vasp import read_vasp, create_FORCE_CONSTANTS, write_supercells_with_displacements
from phonopy import PhonopyGruneisen

@explicit_serialize
class CheckOptimization(FiretaskBase):
    """
    Check if structure optimization is undergoing fully. The critia is the ionic step
    in OSZICAR should be 1. if not, copy CONTCAR to POSCAR and optimize again.
    
    Required params:
        (none)
    
    Optional params:
        (As for MyOptimizeFW:
            ["vasp_input_set", "vasp_cmd", "db_file", "name", "count", "kwargs"])      
    """
    
    optional_params = ["vasp_input_set", "vasp_input_set_params", "vasp_cmd", "db_file", "name", "count", "kwargs"]
    _fw_name = "Check Optimization"

    def run_task(self, fw_spec):
        try:
            OSZICAR = Oszicar('OSZICAR')
        except VaspParserError as err:
            print(err)
        else:
            if len(OSZICAR.ionic_steps) == 1:
                new_name = "{}-{}".format(self.get("name"), "final")
                calc_locs = list(fw_spec.get("calc_locs", []))
                calc_locs.append({"name": new_name,
                                  "filesystem": env_chk(self.get('filesystem', None), fw_spec),
                                  "path": self.get("path", os.getcwd())})
                return FWAction(mod_spec=[{'_push_all': {'calc_locs': calc_locs}}])
            else:
                stru = Structure.from_file("CONTCAR")
                kpoint_set = Generate_kpoints(stru, 0.03)
                if self.get("vasp_input_set", None) is not None:
                    vasp_input_set_temp = self.get("vasp_input_set")
                    tempdict = dict(vasp_input_set_temp.incar.items())
                    vasp_input_set = MPRelaxSet(stru, user_incar_settings=tempdict, user_kpoints_settings=kpoint_set)
                else:
                    vasp_input_set_params = self.get("vasp_input_set_params")
                    vasp_input_set = MPRelaxSet(stru, user_incar_settings=vasp_input_set_params, user_kpoints_settings=kpoint_set)
                
                vasp_cmd = self.get("vasp_cmd")
                db_file = self.get("db_file")
                name = self.get("name")
                count = self.get("count")
                kwargs = self.get("kwargs", {})
                calc_locs = list(fw_spec.get("calc_locs", []))
                calc_locs.append({"name": "{}-{}".format(name, str(count)),
                                  "filesystem": env_chk(self.get('filesystem', None), fw_spec),
                                  "path": self.get("path", os.getcwd())})
                count = count + 1
                from aicon.myfireworks import MyOptimizeFW
                new_fw = MyOptimizeFW(structure=stru, vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                                    db_file=db_file, name=name, count=count, **kwargs)
                
                return FWAction(mod_spec=[{'_push_all': {'calc_locs': calc_locs}}], detours=new_fw)

@explicit_serialize
class WriteVaspStaticFromPrev(FiretaskBase):
    """
    Writes input files for a static run. Assumes that output files from a previous
    (e.g., optimization) run can be accessed in current dir or prev_calc_dir. Also allows
    lepsilon (dielectric constant) calcs.

    Required params:
        (none)

    Optional params:
        (documentation for all other optional params can be found in MPStaticSet)
    """

    optional_params = ["prev_calc_dir", "user_incar_settings", "user_kpoints_settings",
                       "standardize", "sym_prec", "lepsilon", "supercell", "other_params"]
    _fw_name = "Write Static"
    
    def run_task(self, fw_spec):
        unitcell = Structure.from_file("POSCAR")
        supercell = self.get("supercell", None)
        if supercell is not None:
            os.system('cp POSCAR POSCAR-unitcell')
            unitcell.make_supercell(supercell)

        lepsilon = self.get("lepsilon", False)
        standardize = self.get("standardize", False)        
        other_params = self.get("other_params", {})
        user_incar_settings = self.get("user_incar_settings", {})
        finder = SpacegroupAnalyzer(unitcell)
        prims = finder.get_primitive_standard_structure()
        # for lepsilon runs, the kpoints should be denser
        if lepsilon:
            kpoint_set = Generate_kpoints(prims, 0.02)
            struct = prims
        elif standardize:
            kpoint_set = Generate_kpoints(prims, 0.03)
            struct = prims
        else:
            kpoint_set = Generate_kpoints(unitcell, 0.03)
            struct = unitcell

        vis = MPStaticSet(struct,
                          user_incar_settings=user_incar_settings,
                          user_kpoints_settings=kpoint_set,
                          sym_prec=self.get("sym_prec", 0.1),
                          lepsilon=lepsilon, **other_params)
        vis.write_input(".")


@explicit_serialize
class WriteVaspNSCFFromPrev(FiretaskBase):
    """
    Writes input files for an NSCF static run. Assumes that output files from an
    scf job can be accessed. There are many options, e.g. uniform mode,
    line mode, adding the optical properties, etc.

    Required params:
        (none)

    Optional params:
        (documentation for all optional params can be found in NonSCFVaspInputSet)
    """

    required_params = []
    optional_params = ["prev_calc_dir", "copy_chgcar", "nbands_factor", "reciprocal_density",
                       "kpoints_line_density", "small_gap_multiply", "standardize", "sym_prec",
                       "mode", "other_params", "user_incar_settings"]
    _fw_name = "Write NSCF"
    
    def run_task(self, fw_spec):
        vis = MPNonSCFSet.from_prev_calc(
            prev_calc_dir=self.get("prev_calc_dir", "."),
            copy_chgcar=self.get("copy_chgcar", False),
            user_incar_settings=self.get("user_incar_settings", {}),
            nbands_factor=self.get("nbands_factor", 2),
            reciprocal_density=self.get("reciprocal_density", 100),
            kpoints_line_density=self.get("kpoints_line_density", 30),
            small_gap_multiply=self.get("small_gap_multiply", None),
            standardize=self.get("standardize", False),
            sym_prec=self.get("sym_prec", 0.1),
            mode=self.get("mode", "line"),
            **self.get("other_params", {}))
        vis.write_input(".")


@explicit_serialize
class WriteEMCInput(FiretaskBase):
    """
    Write INPCAR file for emc and KPOINTS.
    
    Required params:
        (["bnd_name", "calc_loc"])

    Optional params:
        (["step_size"])
    """
    
    required_params = ["bnd_name", "calc_loc"]
    optional_params = ["step_size"]
    _fw_name = "Write EMCInput"
    
    def run_task(self, fw_spec):
        calc_loc = get_calc_loc(self["calc_loc"],fw_spec["calc_locs"]) if self.get("calc_loc") else {}
        filepath = calc_loc['path']
        
        C = Electron()
        C.Get_bandstru(filepath)
        
        if self["bnd_name"] == "CBM":
            coord = C.engband.get_cbm()['kpoint'].frac_coords
            bnd_num = np.min(C.engband.get_cbm()['band_index'][Spin.up]) + 1
        elif self["bnd_name"] == "VBM":
            coord = C.engband.get_vbm()['kpoint'].frac_coords
            bnd_num = np.max(C.engband.get_vbm()['band_index'][Spin.up]) + 1
        elif self["bnd_name"] == "CSB":
            C.Get_SB()
            if hasattr(C, 'CSB'):
                coord = C.engband.kpoints[C.CSB.pos["kptindex"]].frac_coords
                bnd_num = np.min(C.engband.get_cbm()['band_index'][Spin.up]) + 1
            else:
                return FWAction(exit=True)
        elif self["bnd_name"] == "VSB":
            C.Get_SB()
            if hasattr(C, 'VSB'):
                coord = C.engband.kpoints[C.VSB.pos["kptindex"]].frac_coords
                bnd_num = np.max(C.engband.get_vbm()['band_index'][Spin.up]) + 1
            else:
                return FWAction(exit=True)
        else:
            raise ValueError("Must specify bnd_name")
        
        lattice = C.engband.structure.lattice.matrix
        
        Write_INPCAR(coord, self["step_size"], bnd_num, "V", lattice)
        
        EMC = EffectMass()
        inpcar_fh = open('INPCAR', 'r')
        (kpt, stepsize, band, prg, basis) = EMC.parse_inpcar(inpcar_fh)
        EMC.get_kpointsfile(kpt, stepsize, prg, basis)
        

@explicit_serialize
class WriteVaspForDeformedCrystal(FiretaskBase):
     """
     Overwrite INCAR and POSCAR for a structure optimization for deformed crystal. 
     Assumes that output files from a previous (e.g., optimization) run can be accessed in current dir or prev_calc_dir. 

     Required params:
        (["strain", "user_incar_settings"])

     Optional params:
        (None)
     """
        
     required_params = ["strain", "user_incar_settings"]
     _fw_name = "Write Vasp For Deformed Crystal"
     
     def run_task(self, fw_spec):
         strain =  self.get("strain", 0.0)
         user_incar_settings = self.get("user_incar_settings", {})
         
         struct = Structure.from_file("POSCAR")
         struct.apply_strain(strain)
         INCAR = Incar(user_incar_settings)
         
         struct.to(filename="POSCAR")
         INCAR.write_file("INCAR")
         
         
@explicit_serialize
class WritePhononBand(FiretaskBase):
    """
    Write 2nd FORCE CONSTANT file and band.yaml file. 
    
    Required params:
        (["supercell"])
        
    Optional params:
        (None)        
    """
    
    _fw_name = "Write Phonon Band"
    required_params = ["supercell"]
    
    def run_task(self, fw_spec):
        create_FORCE_CONSTANTS("vasprun.xml", False, 1)
        (Keylist,Coordslist,prims,transmat) = get_highsympath("POSCAR-unitcell")
        phonon = phonopy.load(supercell_matrix=self.get("supercell"),
                              primitive_matrix=transmat,
                              unitcell_filename="POSCAR-unitcell",
                              calculator="vasp",
                              is_nac=False,
                              force_constants_filename="FORCE_CONSTANTS")
        points = get_band_qpoints([np.array(Coordslist)], 51)
        phonon.run_band_structure(points, with_group_velocities=True)
        phonon.write_yaml_band_structure()


@explicit_serialize
class BuildAICONDir(FiretaskBase):
    """
    Build the directory for AICON calculation, the name of each subdirectory is specific
    and should not be changed. 
    
    Required params:
        (None)
        
    Optional params:
        (None)
    """
    
    _fw_name = "Build AICON Directory"
    
    def run_task(self, fw_spec):
        files_to_copy = ["POSCAR", "INCAR", "KPOINTS", "vasprun.xml", "OUTCAR", "POTCAR"]
        files_to_copy_add = ["POSCAR", "INCAR", "KPOINTS", "vasprun.xml", "OUTCAR", "INPCAR", "EIGENVAL"]
        path_dict = dict()
        calc_loc_equi = get_calc_loc("equi nscf",fw_spec["calc_locs"])
        path_dict["equi"]=calc_loc_equi["path"]
        calc_loc_05 = get_calc_loc("0.5per nscf",fw_spec["calc_locs"])
        path_dict["0.5per"]=calc_loc_05["path"]
        calc_loc_10 = get_calc_loc("1.0per nscf",fw_spec["calc_locs"])
        path_dict["1.0per"]=calc_loc_10["path"]
        calc_loc_dielect = get_calc_loc("dielectric",fw_spec["calc_locs"])
        path_dict["dielect"]=calc_loc_dielect["path"]
        calc_loc_elastic = get_calc_loc("elastic",fw_spec["calc_locs"])
        path_dict["elastic"]=calc_loc_elastic["path"]
        calc_loc_CBM = get_calc_loc("CBM",fw_spec["calc_locs"])
        path_dict["CBM"]=calc_loc_CBM["path"]
        calc_loc_VBM = get_calc_loc("VBM",fw_spec["calc_locs"])
        path_dict["VBM"]=calc_loc_VBM["path"]
        try:
            calc_loc_CSB = get_calc_loc("CSB",fw_spec["calc_locs"])
        except ValueError as err:
            print(err)
        else:
            path_dict["CSB"]=calc_loc_CSB["path"]
        try:
            calc_loc_VSB = get_calc_loc("VSB",fw_spec["calc_locs"])
        except ValueError as err:
            print(err)
        else:
            path_dict["VSB"]=calc_loc_VSB["path"]
            
        curr_dir = os.getcwd()
        for key, path in path_dict.items():
            new_dir = os.path.join(curr_dir, key)
            os.makedirs(new_dir)
            if key not in ["CBM", "VBM", "CSB", "VSB"]:
                copy = CopyFiles(from_dir=path, to_dir=new_dir, files_to_copy=files_to_copy)
                copy.run_task(fw_spec)
            else:
                copy = CopyFiles(from_dir=path, to_dir=new_dir, files_to_copy=files_to_copy_add)
                copy.run_task(fw_spec)
        
        
@explicit_serialize
class BuildPhonopyDir(FiretaskBase):
    """
    Build the directory for gruneisen parameters calculation, the name of each subdirectory is specific
    and should not be changed. 
    
    Required params:
        (["supercell"])
        
    Optional params:
        (None)
    """
    
    _fw_name = "Build Phonopy Directory"
    required_params = ["supercell"]
    
    def run_task(self, fw_spec):
        files_to_copy = ["POSCAR-unitcell", "INCAR", "vasprun.xml", "FORCE_CONSTANTS", "band.yaml"]
        path_dict = dict()
        calc_loc_orig = get_calc_loc("orig phonon band",fw_spec["calc_locs"])
        path_dict["orig"]=calc_loc_orig["path"]
        calc_loc_minus = get_calc_loc("minus phonon band",fw_spec["calc_locs"])
        path_dict["minus"]=calc_loc_minus["path"]
        calc_loc_plus = get_calc_loc("plus phonon band",fw_spec["calc_locs"])
        path_dict["plus"]=calc_loc_plus["path"]
        
        curr_dir = os.getcwd()
        for key, path in path_dict.items():
            new_dir = os.path.join(curr_dir, key)
            os.makedirs(new_dir)
            copy = CopyFiles(from_dir=path, to_dir=new_dir, files_to_copy=files_to_copy)
            copy.run_task(fw_spec)
            
        phonons = {}
        for vol in ("orig", "plus", "minus"):
            (Keylist,Coordslist,prims,transmat) = get_highsympath("%s/POSCAR-unitcell" % vol)
            phonon = phonopy.load(supercell_matrix=self.get("supercell"),
                                  primitive_matrix=transmat,
                                  unitcell_filename="%s/POSCAR-unitcell" % vol,
                                  calculator="vasp",
                                  is_nac=False,
                                  force_constants_filename="%s/FORCE_CONSTANTS" % vol)
            phonons[vol] = phonon
        
        gruneisen = PhonopyGruneisen(phonons["orig"],
                                     phonons["plus"],
                                     phonons["minus"])
        bands=get_band_qpoints([np.array(Coordslist)], 51)
        gruneisen.set_band_structure(bands)
        gruneisen.write_yaml_band_structure()


@explicit_serialize
class RunAICONForElec(FiretaskBase):
    """
    Run AICON to calculate electrical conductivity.
    
    Required params:
        (["mode", "Temp", "Doping", "ifSB"])
        
    """
    
    required_params = ["mode", "Temp", "Doping", "ifSB"]
    _fw_name = "Run AICON For Elec"
    
    def run_task(self, fw_spec):         
         mode = self.get("mode", "standard")
         Temp = self.get("Temp", [300])
         Doping = self.get("Doping", [1e19])
         ifSB = self.get("ifSB", True)
         
         Get_Electron("./", Temp, Doping, mode, ifSB)         


@explicit_serialize
class RunAICONForPhon(FiretaskBase):
    """
    Run AICON to calculate lattice thermal conductivity.
    
    Required params:
        (["Temp", "ifscale"])
        
    """
    
    required_params = ["Temp", "ifscale"]
    _fw_name = "Run AICON For Phon"
    
    def run_task(self, fw_spec):
        Temp = self.get("Temp", [300])
        os.system('cp orig/band.yaml ./')
        os.system('cp orig/POSCAR-unitcell ./POSCAR')
        ifscale = self.get("ifscale", False)
        Get_Phonon("./", Temp, ifscale)


@explicit_serialize
class WriteSupercellWithDisp(FiretaskBase):
    """
    Write Supercells with displacement, combine with finite difference method.
    
    Required params:
        (["supercell"])
        
    """
    
    _fw_name = "Write Supercell With Displacement"
    required_params = ["supercell"]
    
    def run_task(self, fw_spec):
        unitcell = read_vasp("POSCAR-unitcell")
        phonon = phonopy.Phonopy(unitcell, self.get("supercell"))
        
        supercell = phonon.get_supercell()
        phonon.generate_displacements()
        supercells = phonon.supercells_with_displacements
        ids = np.arange(len(supercells)) + 1
        write_supercells_with_displacements(supercell, supercells, ids)
        units = get_default_physical_units("vasp")
        phpy_yaml = PhonopyYaml(physical_units=units,
                                settings={'force_sets': False,
                                          'born_effective_charge': False,
                                          'dielectric_constant': False,
                                          'displacements': True})
        phpy_yaml.set_phonon_info(phonon)
        with open("phonopy_disp.yaml", 'w') as w:
            w.write(str(phpy_yaml))
