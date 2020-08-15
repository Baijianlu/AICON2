# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:34:02 2020

@author: Tao.Fan
"""
from datetime import datetime
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from fireworks import Firework, Workflow
from myfireworks import MyOptimizeFW, MyStaticFW, MyNonSCFFW, MyDFPTFW, \
    MyElasticFW, MyEffectivemassFW, CalElecCondFW

def wf_electron_conductivity(structure, vasp_input_set_relax=None, vasp_input_set_static=None, vasp_input_set_band=None, 
                             vasp_input_set_diel=None, vasp_input_set_elastic=None, vasp_kpoint_set=None, vasp_cmd="vasp",
                             db_file=None, mode="standard", Temp=None, Doping=None):
    '''This workflow aims to calculate electronic conductivity of the structure'''
    tag = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    fws = []
    # get the input set for the optimization and update it if we passed custom settings
    vis_relax = MPRelaxSet(structure, user_incar_settings=vasp_input_set_relax, user_kpoints_settings=vasp_kpoint_set)

    # Structure optimization firework
    fw_opt_equi = MyOptimizeFW(structure=structure, vasp_input_set=vis_relax, vasp_cmd=vasp_cmd,
                      db_file=db_file, name="equi structure optimization", count=1)            #通过定义keywds来修改queue的一些参数
    fws.append(fw_opt_equi)       #1
    # static calculations firework
    fw_static_equi = MyStaticFW(structure=structure, vasp_input_set_params=vasp_input_set_static, prev_calc_loc="equi structure optimization-final",
                           vasp_cmd=vasp_cmd, db_file=db_file, name="equi static", parents=fws[0])
    fws.append(fw_static_equi)        #2
    
    # Structure optimization firework for 0.5% larger and 1% larger structures
    vasp_input_set_relax["ISIF"] = 4
    fw_opt_05 = MyOptimizeFW(structure=structure, vasp_input_set_params=vasp_input_set_relax, strain=0.005, prev_calc_loc="equi structure optimization-final", 
                             vasp_cmd=vasp_cmd, db_file=db_file, name="0.5per structure optimization", count=1, parents=fws[0])
    fw_opt_10 = MyOptimizeFW(structure=structure, vasp_input_set_params=vasp_input_set_relax, strain=0.01, prev_calc_loc="equi structure optimization-final", 
                             vasp_cmd=vasp_cmd, db_file=db_file, name="1.0per structure optimization", count=1, parents=fws[0])
    fws.append(fw_opt_05)        #3
    fws.append(fw_opt_10)        #4
    
    fw_static_05 = MyStaticFW(structure=structure, vasp_input_set_params=vasp_input_set_static, prev_calc_loc="0.5per structure optimization-final",
                           vasp_cmd=vasp_cmd, db_file=db_file, name="0.5per static", parents=fws[2])
    fw_static_10 = MyStaticFW(structure=structure, vasp_input_set_params=vasp_input_set_static, prev_calc_loc="1.0per structure optimization-final",
                           vasp_cmd=vasp_cmd, db_file=db_file, name="1.0per static", parents=fws[3])
    fws.append(fw_static_05)     #5
    fws.append(fw_static_10)     #6
    # band structure calculation firework
    fw_band_equi = MyNonSCFFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="equi static", 
                               name="equi nscf", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[1])
    fw_band_05 = MyNonSCFFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="0.5per static", 
                               name="0.5per nscf", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[4])
    fw_band_10 = MyNonSCFFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="1.0per static", 
                               name="1.0per nscf", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[5])
    fws.append(fw_band_equi)    #7
    fws.append(fw_band_05)      #8
    fws.append(fw_band_10)      #9
    # elastic constant calculation
    fw_elastic = MyElasticFW(structure=structure, vasp_input_set_params=vasp_input_set_elastic, prev_calc_loc="equi static",
                             name="elastic", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[1])
    fws.append(fw_elastic)      #10
    # dielect constant calculation
    fw_dielect = MyDFPTFW(structure=structure, user_incar_settings=vasp_input_set_diel, lepsilon=True, prev_calc_loc="equi static",
                          name="dielectric", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[1])
    fws.append(fw_dielect)      #11
    # effective mass
    fw_effectivemass_CBM = MyEffectivemassFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="equi static",
                                             whichbnd="CBM", stepsize=0.01, name="CBM", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[6])
    fw_effectivemass_VBM = MyEffectivemassFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="equi static",
                                             whichbnd="VBM", stepsize=0.01, name="VBM", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[6])
    fw_effectivemass_CSB = MyEffectivemassFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="equi static",
                                             whichbnd="CSB", stepsize=0.01, name="CSB", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[6])
    fw_effectivemass_VSB = MyEffectivemassFW(structure=structure, vasp_input_set_params=vasp_input_set_band, prev_calc_loc="equi static",
                                             whichbnd="VSB", stepsize=0.01, name="VSB", vasp_cmd=vasp_cmd, db_file=db_file, parents=fws[6])
    fws.append(fw_effectivemass_CBM)    #12
    fws.append(fw_effectivemass_VBM)    #13
    fws.append(fw_effectivemass_CSB)    #14
    fws.append(fw_effectivemass_VSB)    #15
   
    # Call AICON
    fw_eleccond = CalElecCondFW(structure=structure, name="electrical conductivity", mode=mode, Temp=Temp, Doping=Doping,
                                db_file=db_file, parents=fws[0:14], spec={"_queueadapter": {"job_name": 'AICON'}})
    fws.append(fw_eleccond) 
    # finally, create the workflow
    wf_electron_conductivity = Workflow(fws)
    wf_electron_conductivity.name = "{}:{}".format(structure.composition.reduced_formula, "electronic transport properties")
    
    return wf_electron_conductivity
