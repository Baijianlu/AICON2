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
from numpy.linalg import inv
import scipy.constants
import pymatgen as pmg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Kpoints

planck = scipy.constants.h
Boltzm = scipy.constants.Boltzmann
MassFluct = {'H':1.1460e-4, 'He':8.3232e-8, 'Li':14.58e-4, 'Be':0.0, 'B':13.54e-4, 'C':7.38695e-05, 'N':1.8577e-05, 'O':3.3590e-05, 'F':0.0, 'Ne':8.2792e-4, 
             'Na':0.0, 'Mg':7.3989e-4, 'Al':0.0, 'Si':2.01222e-4, 'P':0.0, 'S':1.6808e-4, 'Cl':5.8237e-4, 'Ar':3.50987e-05,
             'K':1.64003e-4, 'Ca':2.9756e-4, 'Sc':0.0, 'Ti':2.8645e-4, 'V':9.5492e-07, 'Cr':1.3287e-4, 'Mn':0.0, 'Fe':8.2444e-05, 'Co':0.0, 'Ni':4.3071e-4, 'Cu':2.10858e-4, 'Zn':5.9594e-4, 'Ga':1.9713e-4, 'Ge':5.87597e-4, 'As':0.0, 'Se':4.6268e-4, 'Br':1.56275e-4, 'Kr':2.4849e-4,
             'Rb':1.0969e-4, 'Sr':6.0994e-05, 'Y':0.0, 'Zr':3.42626e-4, 'Nb':0.0, 'Mo':5.9793e-4, 'Tc':0.0, 'Ru':4.0663e-4, 'Rh':0.0, 'Pd':3.0945e-4, 'Ag':8.5796e-05, 'Cd':2.7161e-4, 'In':1.2456e-05, 'Sn':3.34085e-4, 'Sb':6.6075e-05, 'Te':2.8395e-4, 'I':0, 'Xe':2.6779e-4,
             'Cs':0.0, 'Ba':6.2368e-05, 'La':4.7603e-08, 'Ce':2.2495e-05, 'Pr':0.0, 'Nd':2.3159e-4, 'Pm':0.0, 'Sm':3.3472e-4, 'Eu':4.32889e-05, 'Gd':1.27677e-4, 'Tb':0.0, 'Dy':5.20756e-05, 'Ho':0.0, 'Er':7.2459e-05, 'Tm':0.0, 'Yb':8.5449e-05, 'Lu':8.2759e-07, 'Hf':5.2536e-05,
             'Ta':3.80667e-09, 'W':6.9669e-05, 'Re':2.7084e-05,'Os':7.4520e-05, 'Ir':2.5378e-05, 'Pt':3.39199e-05, 'Au':0.0, 'Hg':6.5260e-05, 'Tl':1.99668e-05, 'Pb':1.94476e-05, 'Bi':0.0}
    
def Generate_kpoints(struct, kppa):
    ''' 
    Gererate KPOINTS file with desired grid resolution.
    
    Parameters:
    ----------
    struct: pmg.core.structure object
    kppa: float
        The grid resolution in the reciprocal space, the unit is A-1. 
    '''
    
    comment = "Kpoints with grid resolution = %.3f / A-1" % (kppa)
    recip_lattice = np.array(struct.lattice.reciprocal_lattice.abc)/(2*np.pi)
    num_div = [int(round(l / kppa)) for l in recip_lattice]
    # ensure that numDiv[i] > 0
    num_div = [i if i > 0 else 1 for i in num_div]
    # VASP documentation recommends to use even grids for n <= 8 and odd
    # grids for n > 8.
    num_div = [i + i % 2 if i <= 8 else i - i % 2 + 1 for i in num_div]

    style = Kpoints.supported_modes.Gamma
    num_kpts = 0

    return Kpoints(comment, num_kpts, style, [num_div], [0, 0, 0])

def get_highsympath(filename):
    ''' Get the high symmetry path of phonon spectrum. '''
    
    struct = pmg.core.Structure.from_file(filename)       
    finder = SpacegroupAnalyzer(struct)
    prims = finder.get_primitive_standard_structure()
    HKpath = HighSymmKpath(struct)
    Keys = list()
    Coords = list()
    
    for key in HKpath.kpath['kpoints']:
        Keys.append(key)
        Coords.append(HKpath.kpath['kpoints'][key])
        
    count = 0
    Keylist = list()
    Coordslist = list()
    
    for i in np.arange(len(Keys) - 1):
        if (count-1)%3 == 0:                                                          #count-1 can be intergely divided by 3
            Keylist.append(Keys[0])
            Coordslist.append(Coords[0])
            count+=1
            
        Keylist.append(Keys[i+1])
        Coordslist.append(Coords[i+1])
        count+=1
        
    if (count-1)%3 == 0:
        Keylist.append(Keys[0])
        Coordslist.append(Coords[0])
            
    print('Please set \"BAND\" parameter of phonopy as this:%s' % os.linesep)
    for coord in Coordslist:
        print('%.4f %.4f %.4f  ' % (coord[0], coord[1], coord[2]), end='')
    print('%s' % os.linesep)

    transmat = np.eye(3) 
    if prims.num_sites != struct.num_sites:
        S_T = np.transpose(struct.lattice.matrix)
        P_T = np.transpose(prims.lattice.matrix)
        transmat = inv(S_T) @ P_T
        print('We notice your structure could have a primitive cell. Please set \"PRIMITIVE_AXIS\" parameter of phonopy as this:%s' % os.linesep)
        for coord in transmat:
            print('%.8f %.8f %.8f  ' % (coord[0], coord[1], coord[2]), end='')
        print('%s' % os.linesep)
                
    return Keylist, Coordslist, prims, transmat

def pbc_diff(fcoords1, fcoords2):
    fdist = np.subtract(fcoords1, fcoords2)
    return fdist - np.round(fdist)
    
def get_sym_eq_kpoints(struct, kpoint, cartesian=False, tol=1e-2):
    '''Get the symmetry equivalent kpoints list'''
    
    if not struct:
        return None
    
    sg = SpacegroupAnalyzer(struct)
    symmops = sg.get_point_group_operations(cartesian=cartesian)
    points = np.dot(kpoint, [m.rotation_matrix for m in symmops])
    rm_list = []
    # identify and remove duplicates from the list of equivalent k-points:
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            if np.allclose(pbc_diff(points[i], points[j]), [0, 0, 0], tol):
                rm_list.append(i)
                break
    
    return np.delete(points, rm_list, axis=0)

def get_highsymweight(filename):
    ''' Get the multiplicity of the high symmetry path. '''
    
    struct = pmg.core.Structure.from_file(filename)       
    HKpath = HighSymmKpath(struct)
    Keys = list()
    Coords = list()
        
    for key in HKpath.kpath['kpoints']:
        Keys.append(key)
        Coords.append(HKpath.kpath['kpoints'][key])
    
    Kweight = list()
        
    for i in np.arange(len(Keys)):
        if Keys[i] != '\Gamma':
           Kweight.append(len(get_sym_eq_kpoints(struct, Coords[i]*0.5)))
           
    return Keys, Coords, Kweight

def extract_GV(filepath):
    '''Extract frequency and group velocity information. '''
    
    fp1 = open(filepath + 'band.yaml','r')

    keystr1 = "q-position"
    keystr2 = "distance"
    keystr3 = "frequency"
    keystr4 = "group_velocity"
    keystr5 = "nqpoint:"
    keystr6 = "natom:"
    npoints = 0
    nbands = 0
    countpoints = -1
    countbands = 0
    Gammaflag = 0
    Gamma = list()

    for eachline in fp1:
        eachline = eachline.strip()
        temp = eachline.split()
        if len(temp) > 0:
            if keystr5 == temp[0]:        
                npoints = int(temp[-1])
            elif keystr6 in eachline:
                nbands = int(temp[-1]) * 3
                GroupVec = np.zeros((npoints, nbands+1))
                Frequency = np.zeros((npoints, nbands+1))
            elif keystr1 in eachline:
                countpoints = countpoints + 1
                countbands = 0
                postemp = np.array([np.float(temp[i][:-1]) for i in np.arange(3,6)])
#                print('%f %f %f' % (postemp[0],postemp[1],postemp[2]))
                if postemp[0] == 0.0 and postemp[1] == 0.0 and postemp[2] == 0.0:
                    Gammaflag = 1
            elif keystr2 in eachline:
                #write distance value to the first column of np.array
                GroupVec[countpoints,countbands] = np.float(temp[-1])
                Frequency[countpoints,countbands] = np.float(temp[-1])
                countbands = countbands + 1
                if Gammaflag == 1:
                    Gammaflag = 0
                    if np.float(temp[-1]) not in Gamma:                
                        Gamma.append(np.float(temp[-1]))
            elif keystr3 in eachline:
                Frequency[countpoints,countbands] = np.float(temp[-1])
            elif keystr4 in eachline:
                #write velocity value to the rest colume of each row of np .array
                vectemp = np.array([np.float(temp[i][:-1]) for i in np.arange(2,5)])
                vectemp2 = vectemp**2
                GroupVec[countpoints,countbands] = np.sqrt(vectemp2.sum())
                countbands = countbands + 1
            else:
                continue
        else:
            continue

    fp1.close()
    Gamma = np.array(Gamma)

    return GroupVec,Frequency,Gamma

def extract_GrunP(filepath, nbands=9, npoints=255):
    '''Extract gruneisen parameters information. '''
    
    fp1 = open(filepath + 'gruneisen.yaml','r')

    keystr1 = "q-position"
    keystr2 = "distance"
    keystr3 = "gruneisen"
    keystr4 = "frequency"
    datatype = np.dtype([('freq',np.float),('grun',np.float)])
    Pathpot = np.zeros(npoints)
    GruneisenPara = np.zeros((npoints, nbands),dtype=datatype)
    countpoints = -1
    countbands = 0
        
    for eachline in fp1:
        eachline = eachline.strip()
        temp = eachline.split()
        if len(temp) > 0:
            if keystr1 in eachline:
                countpoints = countpoints + 1
                countbands = 0
            elif keystr2 in eachline:
                #write distance value to the first column of np.array
                Pathpot[countpoints] = np.float(temp[-1])            
            elif keystr3 in eachline:
                #write gruneisen value to the rest colume of each row of np.array
                gruntemp = np.float(temp[-1])            
            elif keystr4 in eachline:
                freqtemp = np.float(temp[-1])
                GruneisenPara[countpoints,countbands] = np.array((freqtemp,gruntemp), dtype=datatype)
                countbands = countbands + 1
            else:
                continue
        else:
            continue
        
    fp1.close()
    
    for j in range(npoints):
        GruneisenPara[j,:] = np.sort(GruneisenPara[j,:], order='freq')
        
    return Pathpot, GruneisenPara

def calc_MGV(filepath,weight):
    '''Calculate branch velocity and frequency. '''
    
    (GroupVec,Frequency,Gamma) = extract_GV(filepath)
    Gamma_index = np.zeros((len(Gamma),2))          
    Bandnum = GroupVec.shape[1] - 1
    Minindex = np.int(0)
    Maxindex = np.int(GroupVec.shape[0] - 1)

    #search groupvec to get Gamma positions
    Pathnum = 0
    for i in np.arange(len(Gamma)):
        Gamma_index[i] = np.array([x_index for x_index,x_value in enumerate(GroupVec[:,0]) if x_value==Gamma[i]])   
        if Gamma_index[i,0] == Gamma_index[i,1]:Pathnum = Pathnum + 1
        else: Pathnum = Pathnum + 2
    
    #the following is for calculating average group velocity of different branch
    modebranch_vel = np.zeros((Bandnum,Pathnum,5))   #the first dimension size equal natom*3
    branch_vel = np.zeros(Bandnum)
    
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(len(Gamma)):                 #
            for k in np.arange(2):
                if k == 0: 
                    if Gamma_index[j,k] > Minindex:
                        modebranch_vel[branch_idx,j*2 + k] = [GroupVec[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k]-6,Gamma_index[j,k]-1)]
                    else:                            #actually, this sentence should never be executed and G point should never be the first point
                        modebranch_vel[branch_idx,j*2 + k] = [GroupVec[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k]+2,Gamma_index[j,k]+7)]
                        break
                if k == 1: 
                    if Gamma_index[j,k] < Maxindex:
                        modebranch_vel[branch_idx,j*2 + k] = [GroupVec[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k]+2,Gamma_index[j,k]+7)]
                    else:
                        break
                
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(Pathnum):
            branch_vel[branch_idx] = branch_vel[branch_idx] + weight[j] * np.average(modebranch_vel[branch_idx,j,:])
        branch_vel[branch_idx] = branch_vel[branch_idx] / np.sum(weight)
    
    #the following is for calculating average frequency of different branch
    modebranch_freq = np.zeros((Bandnum,Pathnum,51))
    branch_freq = np.zeros(Bandnum)
    
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(len(Gamma)): 
            for k in np.arange(2):
                if k == 0:
                    if Gamma_index[j,k] > Minindex:
                        modebranch_freq[branch_idx,j*2 + k] = [Frequency[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k]-50,Gamma_index[j,k]+1)]
                    else:
                        modebranch_freq[branch_idx,j*2 + k] = [Frequency[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k],Gamma_index[j,k]+51)]
                        break
                if k == 1:
                    if Gamma_index[j,k] < Maxindex:
                        modebranch_freq[branch_idx,j*2 + k] = [Frequency[np.int(index),branch_idx+1] for index in np.arange(Gamma_index[j,k],Gamma_index[j,k]+51)]
                    else:
                        break
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(Pathnum):
            branch_freq[branch_idx] = branch_freq[branch_idx] + weight[j] * np.average(modebranch_freq[branch_idx,j,:])
        branch_freq[branch_idx] = branch_freq[branch_idx] / np.sum(weight)    
    
    #the following is for calculating debye temperature for different branch
    branch_DebyeT = np.zeros(Bandnum)
    for branch_idx in np.arange(Bandnum):
        branch_DebyeT[branch_idx] = planck * np.max(Frequency[:,branch_idx+1]) * 1e12/Boltzm
    
    Optic_base = planck * np.min(Frequency[:,4:Bandnum]) * 1e12/Boltzm
    
    return branch_vel,branch_freq,branch_DebyeT,Optic_base

def calc_MGP(filepath,weight):                                     #Gamma:the position of Gamma, weight:multiplicity of Gamma points
    '''Calculate branch gruneisen parameters.'''
    
    (GroupVec,Freq,Gamma) = extract_GV(filepath)
    Gamma_index = np.zeros((len(Gamma),2))          
    Bandnum = GroupVec.shape[1] - 1                 
    Minindex = np.int(0)
    Maxindex = np.int(GroupVec.shape[0] - 1)    
    Path, Gruneisen = extract_GrunP(filepath,Bandnum,GroupVec.shape[0])
    Pathnum = 0
    
    for i in np.arange(len(Gamma)):
        Gamma_index[i] = np.array([x_index for x_index,x_value in enumerate(GroupVec[:,0]) if x_value==Gamma[i]])
        if Gamma_index[i,0] == Gamma_index[i,1]:Pathnum = Pathnum + 1
        else: Pathnum = Pathnum + 2
        
    modebranch_grun = np.zeros((Bandnum,Pathnum,50))              # value for each path in each branch, exclude Gamma point
    branch_grun = np.zeros(Bandnum)                               #average value for different branch
    
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(len(Gamma)): 
            for k in np.arange(2):
                if k == 0:
                    if Gamma_index[j,k] > Minindex:
                        modebranch_grun[branch_idx,j*2 + k] = [Gruneisen[np.int(index),branch_idx]['grun'] for index in np.arange(Gamma_index[j,k]-50,Gamma_index[j,k])]
                    else:
                        modebranch_grun[branch_idx,j*2 + k] = [Gruneisen[np.int(index),branch_idx]['grun'] for index in np.arange(Gamma_index[j,k]+1,Gamma_index[j,k]+51)]
                        break
                if k == 1:
                    if Gamma_index[j,k] < Maxindex:
                        modebranch_grun[branch_idx,j*2 + k] = [Gruneisen[np.int(index),branch_idx]['grun'] for index in np.arange(Gamma_index[j,k]+1,Gamma_index[j,k]+51)]
                    else:
                        break
                    
    for branch_idx in np.arange(Bandnum):
        for j in np.arange(len(weight)):
            branch_grun[branch_idx] = branch_grun[branch_idx] + weight[j] * np.power(np.average(np.abs(modebranch_grun[branch_idx,j,:])),2)
        branch_grun[branch_idx] = np.sqrt(branch_grun[branch_idx] / np.sum(weight))    

    
    return branch_grun

def Get_GVD(filepath):
    '''
    This function is used for obtaining the Gruneisen parameter, group velocity and Debye temperature for kappa calculation, 
    they are all four dimension including three acoustic branches and one "representive" optic branch.
    '''
    gruneisen = np.zeros(4)
    velocity = np.zeros(4)
    DebyeT = np.zeros(4)
    freq = np.zeros(4)
    
    (no1,no2,weight) = get_highsymweight(filepath + "POSCAR")
    (branchvel,branchfreq,branchDebyeT,Optic_base) = calc_MGV(filepath,weight)
    branchgrun = calc_MGP(filepath,weight)
    
    gruneisen[0:3] = branchgrun[0:3]
    velocity[0:3] = branchvel[0:3]
    DebyeT[0:3] = branchDebyeT[0:3]
    freq[0:3] = branchfreq[0:3]
    weightsum = np.sum(branchfreq[3:])
    
    #The following is for optic branch 
    for i in np.arange(3,len(branchfreq)):
        gruneisen[3] = gruneisen[3] + branchfreq[i] * branchgrun[i]
        velocity[3] = velocity[3] + branchfreq[i] * branchvel[i]
        DebyeT[3] = DebyeT[3] + branchfreq[i] * branchDebyeT[i]
    
    gruneisen[3] = gruneisen[3]/weightsum
    velocity[3] = velocity[3]/weightsum
    DebyeT[3] = DebyeT[3]/weightsum
    freq[3] = DebyeT[3] * Boltzm/(1e12 * planck)
    
    return gruneisen, velocity, DebyeT, freq, Optic_base

def calc_MFPS(Elem_tabl):
    '''Calculate mass fluctuation phonon scattering parameter. '''
    
    tab_len  = len(Elem_tabl)
    Mass = [pmg.core.Element[Elem_tabl[i]].atomic_mass for i in np.arange(tab_len)]
    MassSum = np.sum(Mass)
    MFPS = 0.0
    
    for i in np.arange(tab_len):
        MFPS = MFPS + (Mass[i]/MassSum)**2 * MassFluct[Elem_tabl[i]]
    
    MFPS = tab_len * MFPS
    
    return MFPS

def Write_INPCAR(coord, step_size, bnd_num, prg, lattice):
    ''' '''
    fp = open('INPCAR','w')
    fp.write('%.6f %.6f %.6f%s' % (coord[0], coord[1], coord[2], os.linesep))
    fp.write('%f%s' % (step_size, os.linesep))
    fp.write('%d%s' % (bnd_num, os.linesep))
    fp.write('%s%s' % (prg, os.linesep))
    for vector in lattice:
        fp.write('%.9f %.9f %.9f%s' % (vector[0], vector[1], vector[2], os.linesep))
    fp.close()
    return
    
    
    
