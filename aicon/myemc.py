# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:05:25 2019

This file is modified from emc.py

@author: Tao.Fan 
"""
import numpy as np
import sys
import time

class EffectMass(object):
    '''This class is used to calculate band effective mass at specific point and store the data. '''    
    
    EMC_VERSION = '1.51py'
    STENCIL = 5 #3 or 5
    Bohr = 0.52917721092
    
    def __init__(self):
    ###################################################################################################
    #
    #   STENCILS for finite difference
    #
    #   three-point stencil
        self.st3 = []
        self.st3.append([0.0, 0.0, 0.0]); # 0
        self.st3.append([-1.0, 0.0, 0.0]);  self.st3.append([1.0, 0.0, 0.0]);  # dx  1-2
        self.st3.append([0.0, -1.0, 0.0]);  self.st3.append([0.0, 1.0, 0.0])   # dy  3-4
        self.st3.append([0.0, 0.0, -1.0]);  self.st3.append([0.0, 0.0, 1.0])   # dz  5-6
        self.st3.append([-1.0, -1.0, 0.0]); self.st3.append([1.0, 1.0, 0.0]); self.st3.append([1.0, -1.0, 0.0]); self.st3.append([-1.0, 1.0, 0.0]); # dxdy 7-10
        self.st3.append([-1.0, 0.0, -1.0]); self.st3.append([1.0, 0.0, 1.0]); self.st3.append([1.0, 0.0, -1.0]); self.st3.append([-1.0, 0.0, 1.0]); # dxdz 11-14
        self.st3.append([0.0, -1.0, -1.0]); self.st3.append([0.0, 1.0, 1.0]); self.st3.append([0.0, 1.0, -1.0]); self.st3.append([0.0, -1.0, 1.0]); # dydz 15-18
        #
        #   five-point stencil
        self.st5 = []
        self.st5.append([0.0, 0.0, 0.0])
        #
        a = [-2,-1,1,2]
        for i in range(len(a)): #dx
            self.st5.append([float(a[i]), 0., 0.])
        #
        for i in range(len(a)): #dy
            self.st5.append([0., float(a[i]), 0.])
        #
        for i in range(len(a)): #dz
            self.st5.append([0., 0., float(a[i])])
        #
        for i in range(len(a)):
            i1=float(a[i])
            for j in range(len(a)):
                j1=float(a[j])
                self.st5.append([j1, i1, 0.]) # dxdy
        #
        for i in range(len(a)):
            i1=float(a[i])
            for j in range(len(a)):
                j1=float(a[j])
                self.st5.append([j1, 0., i1,]) # dxdz
        #
        for i in range(len(a)):
            i1=float(a[i])
            for j in range(len(a)):
                j1=float(a[j])
                self.st5.append([0., j1, i1]) # dydz

        self.masses = np.zeros(3)
        self.vecs_cart = np.zeros((3,3))
        self.vecs_frac = np.zeros((3,3))
        self.vecs_n = np.zeros((3,3))
    
    def __get__(self, obj, typ = None):
        return self.masses
    
    def __str__(self):
        return '%.3f %.3f %.3f' % (self.masses[0], self.masses[1], self.masses[2])
    
    __repr__ = __str__
    #####################################  Class Method  #####################################################
    
    def MAT_m_VEC(self, m, v):
        p = [ 0.0 for i in range(len(v)) ]
        for i in range(len(m)):
            assert len(v) == len(m[i]), 'Length of the matrix row is not equal to the length of the vector'
            p[i] = sum( [ m[i][j]*v[j] for j in range(len(v)) ] )
        return p
    
    def T(self, m):
        p = [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]
        return p
    
    def N(self, v):
        max_ = 0.
        for item in v:
            if abs(item) > abs(max_): max_ = item
    
        return [ item/max_ for item in v ]
    
    def DET_3X3(self, m):
        assert len(m) == 3, 'Matrix should be of the size 3 by 3'
        return m[0][0]*m[1][1]*m[2][2] + m[1][0]*m[2][1]*m[0][2] + m[2][0]*m[0][1]*m[1][2] - \
               m[0][2]*m[1][1]*m[2][0] - m[2][1]*m[1][2]*m[0][0] - m[2][2]*m[0][1]*m[1][0]
    
    def SCALE_ADJOINT_3X3(self, m, s):
        a = [[0.0 for i in range(3)] for j in range(3)]
    
        a[0][0] = (s) * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        a[1][0] = (s) * (m[1][2] * m[2][0] - m[1][0] * m[2][2])
        a[2][0] = (s) * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    
        a[0][1] = (s) * (m[0][2] * m[2][1] - m[0][1] * m[2][2])
        a[1][1] = (s) * (m[0][0] * m[2][2] - m[0][2] * m[2][0])
        a[2][1] = (s) * (m[0][1] * m[2][0] - m[0][0] * m[2][1])
    
        a[0][2] = (s) * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
        a[1][2] = (s) * (m[0][2] * m[1][0] - m[0][0] * m[1][2])
        a[2][2] = (s) * (m[0][0] * m[1][1] - m[0][1] * m[1][0])
    
        return a
    
    def INVERT_3X3(self, m):
        tmp = 1.0/self.DET_3X3(m)
        return self.SCALE_ADJOINT_3X3(m, tmp)
    
    def IS_SYMMETRIC(self, m):
        for i in range(len(m)):
            for j in range(len(m[i])):
                if m[i][j] != m[j][i]: return False # automatically checks square-shape
    
        return True
    
    def jacobi(self, ainput):
        # from NWChem/contrib/python/mathutil.py
        # possible need to rewrite due to licensing issues
        #
        from math import sqrt
        #
        a = [[ ainput[i][j] for i in range(len( ainput[j] )) ] for j in range(len( ainput )) ] # copymatrix
        n = len(a)
        m = len(a[0])
        if n != m:
            raise 'jacobi: Matrix must be square'
        #
        for i in range(n):
            for j in range(m):
                if a[i][j] != a[j][i]:
                    raise 'jacobi: Matrix must be symmetric'
        #
        tolmin = 1e-14
        tol = 1e-4
        #
        v = [[0.0 for i in range(n)] for j in range(n)] # zeromatrix
        for i in range(n):
            v[i][i] = 1.0
        #
        maxd = 0.0
        for i in range(n):
            maxd = max(abs(a[i][i]),maxd)
        #
        for iter in range(50):
            nrot = 0
            for i in range(n):
                for j in range(i+1,n):
                    aii = a[i][i]
                    ajj = a[j][j]
                    daij = abs(a[i][j])
                    if daij > tol*maxd: # Screen small elements
                        nrot = nrot + 1
                        s = aii - ajj
                        ds = abs(s)
                        if daij > (tolmin*ds): # Check for sufficient precision
                            if (tol*daij) > ds:
                                c = s = 1/sqrt(2.)
                            else:
                                t = a[i][j]/s
                                u = 0.25/sqrt(0.25+t*t)
                                c = sqrt(0.5+u)
                                s = 2.*t*u/c
                            #
                            for k in range(n):
                                u = a[i][k]
                                t = a[j][k]
                                a[i][k] = s*t + c*u
                                a[j][k] = c*t - s*u
                            #
                            for k in range(n):
                                u = a[k][i]
                                t = a[k][j]
                                a[k][i] = s*t + c*u
                                a[k][j]= c*t - s*u
                            #
                            for k in range(n):
                                u = v[i][k]
                                t = v[j][k]
                                v[i][k] = s*t + c*u
                                v[j][k] = c*t - s*u
                            #
                            a[j][i] = a[i][j] = 0.0
                            maxd = max(maxd,abs(a[i][i]),abs(a[j][j]))
            #
            if nrot == 0 and tol <= tolmin:
                break
            tol = max(tolmin,tol*0.99e-2)
        #
        if nrot != 0:
            print('jacobi: [WARNING] Jacobi iteration did not converge in 50 passes!')
        #
        # Sort eigenvectors and values into increasing order
        e = [0.0 for i in range(n)] # zerovector
        for i in range(n):
            e[i] = a[i][i]
            for j in range(i):
                if e[j] > e[i]:
                    (e[i],e[j]) = (e[j],e[i])
                    (v[i],v[j]) = (v[j],v[i])
        #
        return (v,e)
    #
    def cart2frac(self, basis, v):
        return self.MAT_m_VEC( self.T(self.INVERT_3X3(basis)), v )
    
    def fd_effmass_st3(self, e, h):
        m = [[0.0 for i in range(3)] for j in range(3)]
        m[0][0] = (e[1] - 2.0*e[0] + e[2])/h**2
        m[1][1] = (e[3] - 2.0*e[0] + e[4])/h**2
        m[2][2] = (e[5] - 2.0*e[0] + e[6])/h**2
    
        m[0][1] = (e[7] + e[8] - e[9] - e[10])/(4.0*h**2)
        m[0][2] = (e[11] + e[12] - e[13] - e[14])/(4.0*h**2)
        m[1][2] = (e[15] + e[16] - e[17] - e[18])/(4.0*h**2)
    
        # symmetrize
        m[1][0] = m[0][1]
        m[2][0] = m[0][2]
        m[2][1] = m[1][2]
        #
    #    print '-> fd_effmass_st3: Effective mass tensor:\n'
    #    for i in range(len(m)):
    #        print '%15.8f %15.8f %15.8f' % (m[i][0], m[i][1], m[i][2])
    #    print ''
    #    #
        return m
    
    def fd_effmass_st5(self, e, h):
        m = [[0.0 for i in range(3)] for j in range(3)]
        #
        m[0][0] = (-(e[1]+e[4])  + 16.0*(e[2]+e[3])   - 30.0*e[0])/(12.0*h**2)
        m[1][1] = (-(e[5]+e[8])  + 16.0*(e[6]+e[7])   - 30.0*e[0])/(12.0*h**2)
        m[2][2] = (-(e[9]+e[12]) + 16.0*(e[10]+e[11]) - 30.0*e[0])/(12.0*h**2)
        #
        m[0][1] = (-63.0*(e[15]+e[20]+e[21]+e[26]) + 63.0*(e[14]+e[17]+e[27]+e[24]) \
                   +44.0*(e[16]+e[25]-e[13]-e[28]) + 74.0*(e[18]+e[23]-e[19]-e[22]))/(600.0*h**2)
        m[0][2] = (-63.0*(e[31]+e[36]+e[37]+e[42]) + 63.0*(e[30]+e[33]+e[43]+e[40]) \
                   +44.0*(e[32]+e[41]-e[29]-e[44]) + 74.0*(e[34]+e[39]-e[35]-e[38]))/(600.0*h**2)
        m[1][2] = (-63.0*(e[47]+e[52]+e[53]+e[58]) + 63.0*(e[46]+e[49]+e[59]+e[56]) \
                   +44.0*(e[48]+e[57]-e[45]-e[60]) + 74.0*(e[50]+e[55]-e[51]-e[54]))/(600.0*h**2)
        #
        # symmetrize
        m[1][0] = m[0][1]
        m[2][0] = m[0][2]
        m[2][1] = m[1][2]
        #
    #    print '-> fd_effmass_st5: Effective mass tensor:\n'
    #    for i in range(3):
    #        print '%15.8f %15.8f %15.8f' % (m[i][0], m[i][1], m[i][2])
    #    print ''
        #
        return m
    
    def generate_kpoints(self, kpt_frac, st, h, prg, basis):
        from math import pi
        #
        # working in the reciprocal space
        m = self.INVERT_3X3(self.T(basis))
        basis_r = [[ m[i][j]*2.0*pi for j in range(3) ] for i in range(3) ]
        #
        kpt_rec = self.MAT_m_VEC(self.T(basis_r), kpt_frac)
    #    print '-> generate_kpoints: K-point in reciprocal coordinates: %5.3f %5.3f %5.3f' % (kpt_rec[0], kpt_rec[1], kpt_rec[2])
        #
        if prg == 'V' or prg == 'P':
            h = h*(1/EffectMass.Bohr) # [1/A]
        #
        kpoints = []
        for i in range(len(st)):
            k_c_ = [ kpt_rec[j] + st[i][j]*h for j in range(3) ] # getting displaced k points in Cartesian coordinates
            k_f = self.cart2frac(basis_r, k_c_)
            kpoints.append( [k_f[0], k_f[1], k_f[2]] )
        #
        return kpoints
    
    def parse_bands_CASTEP(self, eigenval_fh, band, diff2_size, debug=False):
    
        # Number of k-points X
        nkpt = int(eigenval_fh.readline().strip().split()[3])
    
        # Number of spin components X
        spin_components = float(eigenval_fh.readline().strip().split()[4])
    
        # Number of electrons X.00 Y.00
        tmp = eigenval_fh.readline().strip().split()
        if spin_components == 1:
            nelec = int(float(tmp[3]))
            n_electrons_down = None
        elif spin_components == 2:
            nelec = [float(tmp[3])]
            n_electrons_down = int(float(tmp[4]))
    
        # Number of eigenvalues X
        nband = int(eigenval_fh.readline().strip().split()[3])
    
        energies = []
        # Get eigenenergies and unit cell from .bands file
        while True:
            line = eigenval_fh.readline()
            if not line:
                break
            #
            if 'Spin component 1' in line:
                for i in range(1, nband + 1):
                    energy = float(eigenval_fh.readline().strip())
                    if band == i:
                        energies.append(energy)
    
        return energies
    
    def parse_EIGENVAL_VASP(self, eigenval_fh, band, diff2_size, debug=False):
        ev2h = 1.0/27.21138505
        eigenval_fh.seek(0) # just in case
        eigenval_fh.readline()
        eigenval_fh.readline()
        eigenval_fh.readline()
        eigenval_fh.readline()
        eigenval_fh.readline()
        #
        nelec, nkpt, nband = [int(s) for s in eigenval_fh.readline().split()]
    #    if debug: print 'From EIGENVAL: Number of the valence band is %d (NELECT/2)' % (nelec/2)
        if band > nband:
            print('Requested band (%d) is larger than total number of the calculated bands (%d)!' % (band, nband))
            sys.exit(1)
    
        energies = []
        for i in range(diff2_size):
            eigenval_fh.readline() # empty line
            eigenval_fh.readline() # k point coordinates
            for j in range(1, nband+1):
                line = eigenval_fh.readline()
                if band == j:
                    energies.append(float(line.split()[1])*ev2h)
    
    #    if debug: print ''
        return energies
    #
    def parse_nscf_PWSCF(self, eigenval_fh, band, diff2_size, debug=False):
        ev2h = 1.0/27.21138505
        eigenval_fh.seek(0) # just in case
        engrs_at_k = []
        energies = []
        #
        while True:
            line = eigenval_fh.readline()
            if not line:
                break
            #
            if "End of band structure calculation" in line:
                for i in range(diff2_size):
                    #
                    while True:
                        line = eigenval_fh.readline()
                        if "occupation numbers" in line:
                            break
                        #
                        if "k =" in line:
                            a = [] # energies at a k-point
                            eigenval_fh.readline() # empty line
                            #
                            while True:
                                line = eigenval_fh.readline()
                                if line.strip() == "": # empty line
                                    break
                                #
                                a.extend(line.strip().split())
                            #
                            #print a
                            assert len(a) <= band, 'Length of the energies array at a k-point is smaller than band param'
                            energies.append(float(a[band-1])*ev2h)
        #
        #print engrs_at_k
        return energies
    #
    def parse_inpcar(self, inpcar_fh, debug=False):
        import re
        #
        kpt = []       # k-point at which eff. mass in reciprocal reduced coords (3 floats)
        stepsize = 0.0 # stepsize for finite difference (1 float) in Bohr
        band = 0       # band for which eff. mass is computed (1 int)
        prg = ''       # program identifier (1 char)
        basis = []     # basis vectors in cartesian coords (3x3 floats), units depend on the program identifier
        #
        inpcar_fh.seek(0) # just in case
        p = re.search(r'^\s*(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)', inpcar_fh.readline())
        if p:
            kpt = [float(p.group(1)), float(p.group(2)), float(p.group(3))]
            if debug: print("Found k point in the reduced reciprocal space: %5.3f %5.3f %5.3f" % (kpt[0], kpt[1], kpt[2]))
        else:
            print("Was expecting k point on the line 0 (3 floats), didn't get it, exiting...")
            sys.exit(1)
    
        p = re.search(r'^\s*(\d+\.\d+)', inpcar_fh.readline())
        if p:
            stepsize = float(p.group(1))
            if debug: print("Found stepsize of: %5.3f (1/Bohr)" % stepsize)
        else:
            print("Was expecting a stepsize on line 1 (1 float), didn't get it, exiting...")
            sys.exit(1)
    
        p = re.search(r'^\s*(\d+)', inpcar_fh.readline())
        if p:
            band = int(p.group(1))
            if debug: print("Requested band is : %5d" % band)
        else:
            print("Was expecting band number on line 2 (1 int), didn't get it, exiting...")
            sys.exit(1)
    
        p = re.search(r'^\s*(\w)', inpcar_fh.readline())
        if p:
            prg = p.group(1)
            if debug: print("Program identifier is: %5c" % prg)
        else:
            print("Was expecting program identifier on line 3 (1 char), didn't get it, exiting...")
            sys.exit(1)
    
        for i in range(3):
            p = re.search(r'^\s*(-*\d+\.\d+)\s+(-*\d+\.\d+)\s+(-*\d+\.\d+)', inpcar_fh.readline())
            if p:
                basis.append([float(p.group(1)), float(p.group(2)), float(p.group(3))])
    
        if debug: 
            print("Real space basis:")
            for i in range(len(basis)):
                print('%9.7f %9.7f %9.7f' % (basis[i][0], basis[i][1], basis[i][2]))
    
        if debug: print('')
    
        return kpt, stepsize, band, prg, basis
    
    def get_eff_masses(self, m, basis):
        #
        vecs_cart = [[0.0 for i in range(3)] for j in range(3)]
        vecs_frac = [[0.0 for i in range(3)] for j in range(3)]
        vecs_n    = [[0.0 for i in range(3)] for j in range(3)]
        #
        eigvec, eigval = self.jacobi(m)
        #
        for i in range(3):
            vecs_cart[i] = eigvec[i]
            vecs_frac[i] = self.cart2frac(basis, eigvec[i])
            vecs_n[i]    = self.N(vecs_frac[i])
        #
        em = [ 1.0/eigval[i] for i in range(len(eigval)) ]
        return em, vecs_cart, vecs_frac, vecs_n
    #
    def cal_effmass(self, kpt, stepsize, band, prg, basis, output_fn):
        if EffectMass.STENCIL == 3:
            fd_effmass = self.fd_effmass_st3
            st = self.st3
        elif EffectMass.STENCIL == 5:
            fd_effmass = self.fd_effmass_st5
            st = self.st5
        else:
            print('main: [ERROR] Wrong value for STENCIL, should be 3 or 5.')
            sys.exit(1)
        #
        #
        try:
            output_fh = open(output_fn, 'r')
        except IOError:
            sys.exit("Couldn't open input file "+output_fn+", exiting...\n")
        #
        if output_fn:
            #
            energies = []
            if prg.upper() == 'V' or prg.upper() == 'C':
                energies = self.parse_EIGENVAL_VASP(output_fh, band, len(st))
                m = fd_effmass(energies, stepsize)
            #
            if prg.upper() == 'Q':
                energies = self.parse_nscf_PWSCF(output_fh, band, len(st))
                m = fd_effmass(energies, stepsize)
            #
            if prg.upper() == 'P':
                energies = self.parse_bands_CASTEP(output_fh, band, len(st))
                m = fd_effmass(energies, stepsize)
            #
            masses, vecs_cart, vecs_frac, vecs_n = self.get_eff_masses(m, basis)
            self.vecs_cart = np.array(vecs_cart)
            self.vecs_frac = np.array(vecs_frac)
            self.vecs_n = np.array(vecs_n)
            self.masses = np.array(masses)
        #
        maxindx =np.argmax(np.abs(self.masses))
        temp = 1.0
        for i in np.arange(3):
            if i == maxindx:
                self.parallelmass = self.masses[i]
            else:
                temp = temp * self.masses[i]
        
        self.verticalmass = np.sign(self.masses[0]) * np.sqrt(temp)
        self.condeffmass = 3.0 / (1/self.masses[0] + 1/self.masses[1] + 1/self.masses[2])
        self.doseffmass = np.sign(self.masses[0]) * np.abs(self.masses[0] * self.masses[1] * self.masses[2])**(1/3)
        return 
    
    def get_kpointsfile(self, kpt, stepsize, prg, basis):
        if EffectMass.STENCIL == 3:
            st = self.st3
        elif EffectMass.STENCIL == 5:
            st = self.st5
        else:
            print('main: [ERROR] Wrong value for STENCIL, should be 3 or 5.')
            sys.exit(1)
        
        kpoints = self.generate_kpoints(kpt, st, stepsize, prg, basis)
        kpoints_fh = open('KPOINTS', 'w')
        kpoints_fh.write("generate with stepsize: "+str(stepsize)+"\n")
        kpoints_fh.write("%d\n" % len(st))
        kpoints_fh.write("Reciprocal\n")
        #
        for i, kpt in enumerate(kpoints):
            kpoints_fh.write( '%15.10f %15.10f %15.10f 0.01\n' % (kpt[0], kpt[1], kpt[2]) )
        #
        kpoints_fh.close()
        
        return
