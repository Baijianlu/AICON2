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

from atomate.vasp.config import HALF_KPOINTS_FIRST_RELAX, RELAX_MAX_FORCE, \
    VASP_CMD, DB_FILE
from fireworks import Firework
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, pass_vasp_result
from atomate.vasp.firetasks.parse_outputs import VaspToDb
from atomate.vasp.firetasks.run_calc import RunVaspCustodian, RunVaspDirect
from atomate.vasp.firetasks.write_inputs import WriteVaspFromIOSet
from aicon.myfiretasks import CheckOptimization, WriteVaspStaticFromPrev, WriteVaspNSCFFromPrev, \
    WriteEMCInput, WriteVaspForDeformedCrystal, BuildAICONDir, RunAICONForElec, WritePhononBand, \
    BuildPhonopyDir, RunAICONForPhon, WriteSupercellWithDisp

class MyOptimizeFW(Firework):

    def __init__(self, structure, name="structure optimization",
                 vasp_input_set=None, vasp_input_set_params=None, count=1,
                 vasp_cmd=VASP_CMD, 
                 ediffg=None, db_file=DB_FILE,
                 prev_calc_loc = False, strain=0.0,
                 force_gamma=True, job_type="normal",
                 max_force_threshold=RELAX_MAX_FORCE,
                 auto_npar=">>auto_npar<<",
                 half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX, parents=None,
                 **kwargs):
        """
        Optimize the given structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
            vasp_input_set_params (dict): Parameters in INCAR to override.
            count (int): A counter to record round of structure optimization.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs
            db_file (str): Path to file specifying db credentials to place output parsing.
            prev_calc_loc (bool or str): If true, copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            strain (float): strain executed on structure in each direction of lattice. 
            force_gamma (bool): Force gamma centered kpoint generation
            job_type (str): custodian job type (default "double_relaxation_run")
            max_force_threshold (float): max force on a site allowed at end; otherwise, reject job
            auto_npar (bool or str): whether to set auto_npar. defaults to env_chk: ">>auto_npar<<"
            half_kpts_first_relax (bool): whether to use half the kpoints for the first relaxation
            parents ([Firework]): Parents of this particular Firework.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """

        t = []
        if parents and prev_calc_loc:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspForDeformedCrystal(strain=strain, user_incar_settings=vasp_input_set_params))    
        else:
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, job_type=job_type, auto_npar=auto_npar, gzip_output=False))
        t.append(CheckOptimization(vasp_input_set=vasp_input_set, vasp_input_set_params=vasp_input_set_params, 
                                   vasp_cmd=vasp_cmd, db_file=db_file, name=name, count=count, kwargs=kwargs))
        
        super(MyOptimizeFW, self).__init__(t, parents=parents, name="{}-{}-{}".
                                         format(structure.composition.reduced_formula, name, str(count)),
                                         **kwargs)


class MyStaticFW(Firework):

    def __init__(self, structure=None, name="static", vasp_input_set=None, vasp_input_set_params=None, vasp_kpoint_set=None,
                 vasp_cmd=VASP_CMD, prev_calc_loc=True, prev_calc_dir=None, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, **kwargs):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): Input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Parameters in INCAR to override.
            vasp_kpoint_set: (Kpoint): Kpoint set to use.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=vasp_input_set_params, standardize=True))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=vasp_input_set_params, standardize=True))
        elif structure:
            vasp_input_set = vasp_input_set or MPStaticSet(structure)
            t.append(WriteVaspFromIOSet(structure=structure,
                                        vasp_input_set=vasp_input_set,
                                        vasp_input_params=vasp_input_set_params))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", gzip_output=False))
        t.append(PassCalcLocs(name=name))
        super(MyStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class MyNonSCFFW(Firework):

    def __init__(self, parents=None, prev_calc_dir=None, structure=None,
                 name="nscf", mode="line", vasp_cmd=VASP_CMD,
                 prev_calc_loc=True, db_file=DB_FILE,
                 vasp_input_set_params=None, **kwargs):
        """
        Standard NonSCF Calculation Firework supporting uniform and line modes.

        Args:
            structure (Structure): Input structure - used only to set the name
                of the FW.
            name (str): Name for the Firework.
            mode (str): "uniform" or "line" mode.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            vasp_input_set_params (dict): Parameters in INCAR to override.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        vasp_input_set_params = vasp_input_set_params or {}

        fw_name = "{}-{} {}".format(structure.composition.reduced_formula if
                                    structure else "unknown", name, mode)
        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir,
                                     additional_files=["CHGCAR"]))
        elif parents:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc,
                                     additional_files=["CHGCAR"]))
        else:
            raise ValueError("Must specify previous calculation for NonSCFFW")

        mode = mode.lower()
        if mode == "uniform":
            t.append(WriteVaspNSCFFromPrev(prev_calc_dir=".", mode="uniform", user_incar_settings=vasp_input_set_params))
        else:
            t.append(WriteVaspNSCFFromPrev(prev_calc_dir=".", mode="line", user_incar_settings=vasp_input_set_params))

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", gzip_output=False))
        t.append(PassCalcLocs(name=name))

        super(MyNonSCFFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class MyDFPTFW(Firework):

    def __init__(self, structure=None, prev_calc_dir=None, name="dielectric", vasp_cmd=VASP_CMD,
                 prev_calc_loc=True, lepsilon=True,
                 db_file=DB_FILE, parents=None, user_incar_settings=None, user_kpoints_settings=None,
                 pass_nm_results=False, **kwargs):
        """
         Static DFPT calculation Firework

        Args:
            structure (Structure): Input structure. If prev_calc_loc, used only to set the
                name of the FW.
            name (str): Name for the Firework.
            lepsilon (bool): Turn on LEPSILON to calculate polar properties
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (str or bool): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            user_incar_settings (dict): Parameters in INCAR to override
            user_kpoints_settings (Kpoint): Kpoint set to use.
            pass_nm_results (bool): if true the normal mode eigen vals and vecs are passed so that
                next firework can use it.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        name = "dielectric" if lepsilon else "phonon"

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        user_incar_settings = user_incar_settings or {}
        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(lepsilon=lepsilon, user_incar_settings=user_incar_settings))
        elif parents and prev_calc_loc:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(lepsilon=lepsilon, user_incar_settings=user_incar_settings))
        elif structure:
            vasp_input_set = MPStaticSet(structure, lepsilon=lepsilon, user_kpoints_settings=user_kpoints_settings,
                                         user_incar_settings=user_incar_settings)
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gzip_output=False))

        if pass_nm_results:
            t.append(pass_vasp_result({"structure": "a>>final_structure",
                                       "eigenvals": "a>>normalmode_eigenvals",
                                       "eigenvecs": "a>>normalmode_eigenvecs"},
                                      parse_eigen=True,
                                      mod_spec_key="normalmodes"))

        t.append(PassCalcLocs(name=name))

        super(MyDFPTFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)

class MyElasticFW(Firework):

    def __init__(self, structure=None, name="elastic", vasp_input_set=None, vasp_input_set_params=None, vasp_kpoint_set=None,
                 vasp_cmd=VASP_CMD, prev_calc_loc=True, prev_calc_dir=None, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, **kwargs):
        """
        Elastic tensor calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents) Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Parameters in INCAR to override.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=vasp_input_set_params))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=vasp_input_set_params))
        elif structure:
            vasp_input_set = vasp_input_set or MPStaticSet(structure)
            t.append(WriteVaspFromIOSet(structure=structure,
                                        vasp_input_set=vasp_input_set,
                                        vasp_input_params=vasp_input_set_params))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", gzip_output=False))
        t.append(PassCalcLocs(name=name))
        
        super(MyElasticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class MyEffectivemassFW(Firework):

    def __init__(self, parents=None, prev_calc_dir=None, structure=None,
                 name="effective mass", mode="uniform", whichbnd="CBM", stepsize=0.01, 
                 vasp_cmd=VASP_CMD, prev_calc_loc="True", db_file=DB_FILE,
                 vasp_input_set_params=None, **kwargs):
        """
        Modified NonSCF Calculation Firework supporting uniform and line modes.

        Args:
            structure (Structure): Input structure - used only to set the name of the FW.
            name (str): Name for the Firework.
            mode (str): "uniform" or "line" mode.
            whichbnd (str): specify the name of band to calculate (CBM, VBM, CSB, VSB).
            stepsize (float): stepsize for finite difference in Bohr. Default is 0.01.
            vasp_cmd (str): Command to run vasp. 
            prev_calc_loc (bool or str): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasp_input_set_params (dict): Parameters in INCAR to override.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        vasp_input_set_params = vasp_input_set_params or {}

        fw_name = "{}-{} {}".format(structure.composition.reduced_formula if
                                    structure else "unknown", name, whichbnd)
        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir,
                                     additional_files=["CHGCAR"]))
        elif parents and prev_calc_loc:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc,
                                     additional_files=["CHGCAR"]))
        else:
            raise ValueError("Must specify previous calculation for MyEffectivemassFW")

        mode = mode.lower()

        t.append(WriteVaspNSCFFromPrev(prev_calc_dir=".", mode="line", user_incar_settings=vasp_input_set_params))
        # write INPCAR and KPOINTS
        t.append(WriteEMCInput(bnd_name=whichbnd, calc_loc='equi nscf', step_size=stepsize))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", gzip_output=False))
        t.append(PassCalcLocs(name=whichbnd))

        super(MyEffectivemassFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class CalElecCondFW(Firework):
    def __init__(self, structure=None, name="electrical conductivity", db_file=DB_FILE, 
                 parents=None, mode=None, Temp=None, Doping=None, ifSB=None, **kwargs):
        """
        electrical conductivity calculation firework
        
        Args:
            structure (Structure): Input structure, used only to set the name of the FW.
            name (str): Name for the Firework.
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            mode (str): AICON mode, either standard or doping.
            Temp (list): Temperature value array.
            Doping (list): Doping value array.
            ifSB (bool): if consider the second band's contribution.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
            
        """
        
        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        t = []
        
        t.append(BuildAICONDir())
        t.append(RunAICONForElec(mode=mode, Temp=Temp, Doping=Doping, ifSB=ifSB))
        t.append(PassCalcLocs(name=name))
        
        super(CalElecCondFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class MyPhononFW(Firework):
    
    def __init__(self, structure=None, prev_calc_dir=None, name="phonon band", vasp_cmd=VASP_CMD,
                 prev_calc_loc=True, db_file=DB_FILE, parents=None, user_incar_settings=None, 
                 user_kpoints_settings=None, supercell=None, **kwargs):
        """
        Phonon calculation Firework using DFPT

        Args:
            structure (Structure): Input structure. If prev_calc_loc, used only to set the
                name of the FW.
            name (str): Name for the Firework.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (str or bool): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            user_incar_settings (dict): Parameters in INCAR to override
            user_kpoints_settings (Kpoint): Kpoint set to use.
            supercell (list) size of supercell: 
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=user_incar_settings, standardize=False, supercell=supercell))
        elif parents and prev_calc_loc:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=user_incar_settings, standardize=False, supercell=supercell))
        elif structure:
            vasp_input_set = MPStaticSet(structure, user_kpoints_settings=user_kpoints_settings,
                                         user_incar_settings=user_incar_settings)
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, gzip_output=False))
        t.append(WritePhononBand(supercell=supercell))
        t.append(PassCalcLocs(name=name))

        super(MyPhononFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class CalPhonCondFW(Firework):
    def __init__(self, structure=None, name="thermal conductivity", db_file=DB_FILE, 
                 parents=None, Temp=None, ifscale=None, supercell=None, **kwargs):
        """
        lattice thermal conductivity calculation firework
        
        Args:
            structure (Structure): Input structure, used only to set the name of the FW.
            name (str): Name for the Firework.
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            Temp (list): Temperature value array.
            ifscale (bool): If multiply a scaling factor with Kappa.
            supercell (list): size of supercell.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
            
        """
        
        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        t = []
        
        t.append(BuildPhonopyDir(supercell=supercell))
        t.append(RunAICONForPhon(Temp=Temp, ifscale=ifscale))
        t.append(PassCalcLocs(name=name))
        
        super(CalPhonCondFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class MyPhononFiniteDiffFW(Firework):
    
    def __init__(self, structure=None, prev_calc_dir=None, name="phonon band", vasp_cmd=VASP_CMD,
                 prev_calc_loc=True, db_file=DB_FILE, parents=None, user_incar_settings=None, 
                 user_kpoints_settings=None, supercell=None, **kwargs):
        """
        Phonon calculation Firework using finite difference method

        Args:
            structure (Structure): Input structure. If prev_calc_loc, used only to set the
                name of the FW.
            name (str): Name for the Firework.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (str or bool): Whether to copy outputs from previous run. Defaults to True.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework.
                FW or list of FWS.
            user_incar_settings (dict): Parameters in INCAR to override
            user_kpoints_settings (Kpoint): Kpoint set to use.
            supercell (list) size of supercell: 
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        
        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        t = []

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=user_incar_settings, standardize=False, supercell=supercell))
        elif parents and prev_calc_loc:
            t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspStaticFromPrev(user_incar_settings=user_incar_settings, standardize=False, supercell=supercell))
        elif structure:
            vasp_input_set = MPStaticSet(structure, user_kpoints_settings=user_kpoints_settings,
                                         user_incar_settings=user_incar_settings)
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(WriteSupercellWithDisp(supercell=supercell))
        t.append(PassCalcLocs(name=name))

        super(MyPhononFiniteDiffFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)
        
        
        
        
